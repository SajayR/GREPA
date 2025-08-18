#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEPA-min (one file, readable): reflective prompt evolution loop for classification
-------------------------------------------------------------------------------

Quick start:
  pip install "transformers>=4.43" "torch>=2.1" "openai>=1.30" datasets

Run:
  python gepa_min.py \
    --json_path data/trainset.json \
    --target_label Providing_Guidance \
    --hf_task_model Qwen/Qwen2.5-0.5B-Instruct \
    --openai_reflector_model gpt-4o-mini \
    --iterations 40

What it does:
  - loads your JSON (conversation_history + tutor_responses[*].response + .annotation[label])
  - stratified train/dev/test split
  - class-balanced minibatch sampling
  - acceptance loop: (prompt -> outputs -> casebook -> big LLM -> new prompt) with strict A/B on same batch
  - accuracy scoring using <answer> Yes|No|To some extent
  - uses HF model locally for the task model; uses OpenAI model for reflection

Swap to llama.cpp later:
  - keep this file; just replace the HF task model wrapper with your llama-server client
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

# =========================
# Section 0 — types & seed
# =========================

ALLOWED_LABELS: tuple[str, ...] = ("Yes", "No", "To some extent")

@dataclass
class TrainExample:
    input_text: str      # rendered dialogue + candidate tutor reply
    gold_label: str      # one of ALLOWED_LABELS

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =================================
# Section 1 — data loading/flatten
# =================================

def load_raw_conversations(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Top-level JSON must be a list"
    return data

def render_input_text(conversation_history: str, candidate_tutor_response: str) -> str:
    return (
        "You are evaluating whether the following tutor reply identifies the student's mistake "
        "and provides appropriate guidance according to the rubric.\n\n"
        "=== DIALOGUE CONTEXT ===\n"
        f"{conversation_history.strip()}\n\n"
        "=== CANDIDATE TUTOR RESPONSE TO EVALUATE ===\n"
        f"{candidate_tutor_response.strip()}\n"
    )

def flatten_to_examples(
    raw_conversations: List[Dict[str, Any]],
    target_annotation_field: str = "Providing_Guidance"  # or "Mistake_Identification"
) -> Tuple[List[TrainExample], Dict[str, int]]:
    examples: List[TrainExample] = []
    label_counts: Dict[str, int] = defaultdict(int)

    for conv in raw_conversations:
        conv_history = conv.get("conversation_history", "")
        tutor_responses = conv.get("tutor_responses", {})
        if not isinstance(tutor_responses, dict):
            continue

        for _, payload in tutor_responses.items():
            resp_text = (payload.get("response") or "").strip()
            ann = payload.get("annotation", {}) or {}
            gold = ann.get(target_annotation_field)
            if gold is None:
                continue
            canon = (gold or "").strip().lower()
            if canon == "yes":
                gold_label = "Yes"
            elif canon == "no":
                gold_label = "No"
            elif canon == "to some extent":
                gold_label = "To some extent"
            else:
                continue
            if gold_label not in ALLOWED_LABELS:
                continue
            input_text = render_input_text(conv_history, resp_text)
            examples.append(TrainExample(input_text=input_text, gold_label=gold_label))
            label_counts[gold_label] += 1

    return examples, dict(label_counts)

def stratified_split(
    all_examples: List[TrainExample],
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
    test_frac: float = 0.1,
    rng_seed: int = 17
) -> Tuple[List[TrainExample], List[TrainExample], List[TrainExample]]:
    assert abs((train_frac + dev_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1"
    set_all_seeds(rng_seed)
    by_label: Dict[str, List[TrainExample]] = defaultdict(list)
    for ex in all_examples:
        by_label[ex.gold_label].append(ex)
    train: List[TrainExample] = []
    dev:   List[TrainExample] = []
    test:  List[TrainExample] = []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_dev   = int(round(n * dev_frac))
        n_test  = n - n_train - n_dev
        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train+n_dev])
        test.extend(items[n_train+n_dev:])
    random.shuffle(train); random.shuffle(dev); random.shuffle(test)
    return train, dev, test


# =======================================
# Section 2 — task model (HF transformers)
# =======================================

class SimpleHFChat:
    """
    Minimal HF text-generation wrapper. Not a true chat; we format:
      [SYSTEM PROMPT]\n\n[INPUT TEXT]
    and decode greedily-ish with temperature.

    Swap this with your llama.cpp client later without touching the rest.
    """
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, dtype: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto" if device is None else device,
            torch_dtype=dtype if dtype != "auto" else None,
            trust_remote_code=True
        )
        self.streamer = None  # could plug a streamer for debugging

    def generate(self,
                 system_prompt: str,
                 user_texts: List[str],
                 temperature: float = 0.4,
                 top_p: float = 0.95,
                 max_new_tokens: int = 192,
                 repetition_penalty: float = 1.05) -> List[str]:
        from transformers import GenerationConfig
        prompts = [system_prompt.strip() + "\n\n" + txt.strip() for txt in user_texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        gen_cfg = GenerationConfig(
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with self.model.device:
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)
        # take only the generated tail
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # the decode returns whole prompt+completion; clip heuristic:
        results = []
        for full, prefix in zip(decoded, prompts):
            if full.startswith(prefix):
                results.append(full[len(prefix):].strip())
            else:
                results.append(full.strip())
        return results


# ======================================
# Section 3 — parsing + minibatch scoring
# ======================================

def parse_answer_tag(text: str) -> Optional[str]:
    """
    Extract the label inside <answer>...</answer>, normalize to canonical label if possible.
    """
    lo = text.lower()
    if "<answer>" not in lo or "</answer>" not in lo:
        return None
    try:
        start = lo.index("<answer>") + len("<answer>")
        end = lo.index("</answer>", start)
        raw = text[start:end].strip()  # slice original text to keep case
    except Exception:
        return None
    # normalize to our canonical set
    candidate = raw.strip().lower()
    if candidate == "yes":
        return "Yes"
    if candidate == "no":
        return "No"
    if candidate in {"to some extent", "to_some_extent", "partial", "partially"}:
        return "To some extent"
    # allow pipe form: Yes|No|To some extent (pick first token if they echoed)
    for lbl in ALLOWED_LABELS:
        if lbl.lower() in candidate:
            return lbl
    return None

def evaluate_minibatch_once(
    current_system_prompt: str,
    minibatch_examples: List[TrainExample],
    task_model: SimpleHFChat,
    generation_temperature: float = 0.4,
    generation_top_p: float = 0.95,
    generation_max_new_tokens: int = 192,
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """
    Returns:
      raw_outputs: list of model texts
      scores: list of 0/1 accuracies
      trajectories: list of dicts with input_text, gold_label, model_output_full_text, parsed_label
    """
    user_inputs = [ex.input_text for ex in minibatch_examples]
    try:
        raw_outputs = task_model.generate(
            system_prompt=current_system_prompt,
            user_texts=user_inputs,
            temperature=generation_temperature,
            top_p=generation_top_p,
            max_new_tokens=generation_max_new_tokens,
        )
    except Exception as e:
        # If generation fails entirely, return zeros
        raw_outputs = [""] * len(user_inputs)

    scores: List[float] = []
    trajectories: List[Dict[str, Any]] = []
    for ex, out in zip(minibatch_examples, raw_outputs):
        parsed = parse_answer_tag(out)
        is_correct = 1.0 if (parsed == ex.gold_label) else 0.0
        scores.append(is_correct)
        trajectories.append({
            "input_text": ex.input_text,
            "gold_label": ex.gold_label,
            "model_output_full_text": out,
            "parsed_label": parsed
        })
    return raw_outputs, scores, trajectories


# ===========================================
# Section 4 — class-balanced minibatch sampler
# ===========================================

def build_label_index(examples: List[TrainExample]) -> Dict[str, List[int]]:
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        label_to_indices[ex.gold_label].append(idx)
    for lbl in label_to_indices:
        random.shuffle(label_to_indices[lbl])
    return dict(label_to_indices)

class ClassBalancedSampler:
    def __init__(self, label_to_indices: Dict[str, List[int]], global_seed: int = 17):
        self.label_to_indices = {k: list(v) for k, v in label_to_indices.items()}
        self.labels: List[str] = sorted(self.label_to_indices.keys())
        self.cursors: Dict[str, int] = {lbl: 0 for lbl in self.labels}
        random.seed(global_seed)

    def _next_from_label(self, label: str) -> int:
        bucket = self.label_to_indices[label]
        if not bucket:
            raise ValueError(f"No examples for label: {label}")
        i = self.cursors[label] % len(bucket)
        ex_idx = bucket[i]
        self.cursors[label] += 1
        if self.cursors[label] % len(bucket) == 0:
            random.shuffle(bucket)
        return ex_idx

    def sample_indices(self, batch_size: int) -> List[int]:
        k = len(self.labels)
        if k == 0:
            return []
        base = batch_size // k
        leftover = batch_size - base * k
        counts = {lbl: base for lbl in self.labels}
        for j in range(leftover):
            counts[self.labels[j % k]] += 1
        batch: List[int] = []
        for lbl in self.labels:
            for _ in range(counts[lbl]):
                batch.append(self._next_from_label(lbl))
        random.shuffle(batch)
        return batch


# ==========================================
# Section 5 — reflection casebook (examples)
# ==========================================

def _short_feedback(gold: str, pred: Optional[str], had_answer_tag: bool) -> str:
    if not had_answer_tag:
        return "Output is missing the required <answer> tag with exactly one of {Yes, No, To some extent}."
    if gold == "Yes":
        if pred != "Yes":
            return "Reply lacks actionable guidance; add at least one concrete next step aligned to the rubric."
        return "Correct: guidance present and actionable."
    if gold == "No":
        if pred != "No":
            return "Reply does not identify the student’s mistake; first name the specific error before advising."
        return "Correct: mistake identified; inappropriate guidance avoided."
    # gold == "To some extent"
    if pred != "To some extent":
        return "Partial guidance: require explicit mistake identification and at least one concrete next step."
    return "Correct: partial guidance with adequate scaffolding."

def build_reflection_casebook(
    trajectories: List[Dict[str, Any]],
    k_fail: int = 6,
    k_pass: int = 2
) -> List[Dict[str, Any]]:
    fails: List[Dict[str, Any]] = []
    passes: List[Dict[str, Any]] = []
    for t in trajectories:
        gold = t["gold_label"]
        pred = t.get("parsed_label")
        out_text = t["model_output_full_text"]
        had_answer_tag = (pred is not None)
        fb = _short_feedback(gold, pred, had_answer_tag)
        rec = {
            "Inputs": t["input_text"],
            "Generated Outputs": out_text,
            "Feedback": fb,
        }
        if pred == gold and pred is not None:
            passes.append(rec)
        else:
            fails.append(rec)

    selected: List[Dict[str, Any]] = []
    selected.extend(fails[:k_fail])
    selected.extend(passes[:k_pass])
    if len(selected) < (k_fail + k_pass):
        pool = passes if len(fails) >= k_fail else fails
        need = (k_fail + k_pass) - len(selected)
        selected.extend(pool[:need])
    return selected[: (k_fail + k_pass)]


# ==========================================
# Section 6 — reflector (OpenAI Chat wrapper)
# ==========================================

try:
    from openai import OpenAI  # pip install openai>=1.30
except Exception:
    OpenAI = None  # we'll error at construction if used

ALLOWED_LABELS_TEXT = "Yes|No|To some extent"

def _render_reflection_prompt(current_instruction: str, casebook: List[Dict[str, Any]]) -> str:
    def render_item(i: int, rec: Dict[str, Any]) -> str:
        return (
            f"# Example {i}\n"
            f"## Inputs\n{rec['Inputs']}\n\n"
            f"## Model Output\n{rec['Generated Outputs']}\n\n"
            f"## Feedback\n{rec['Feedback']}\n"
        )
    examples_text = "\n\n".join(render_item(i+1, rec) for i, rec in enumerate(casebook))
    return (
        "You are revising an instruction for a classifier that must output EXACTLY one label inside an <answer> tag.\n"
        f"Allowed labels are: {ALLOWED_LABELS_TEXT}\n\n"
        "The classifier should briefly think, then output the label.\n"
        "Here is the current instruction:\n```"
        f"{current_instruction}"
        "```\n\n"
        "Below are examples of inputs, the model's outputs, and feedback on how to improve:\n"
        "```\n"
        f"{examples_text}\n"
        "```\n\n"
        "Write a revised instruction that:\n"
        "  1) Keeps the exact output contract:\n"
        "     <think>…</think>\n"
        f"     <answer> {ALLOWED_LABELS_TEXT} </answer>\n"
        "  2) Emphasizes: identify the student’s mistake before guidance; provide at least one concrete next step when guidance is appropriate; avoid generic praise; avoid hallucinating details.\n"
        "  3) States the allowed labels explicitly.\n\n"
        "Return ONLY the new instruction inside triple backticks.\n"
    )

def _parse_backticked_instructions(text: str) -> Optional[str]:
    if "```" not in text:
        return None
    start = text.find("```") + 3
    end = text.find("```", start)
    if end == -1:
        return None
    new_inst = text[start:end].strip()
    return new_inst if new_inst else None

def _passes_contract(new_instruction: str) -> bool:
    must_have = ["<answer>", "</answer>", "Yes", "No", "To some extent"]
    return all(tok in new_instruction for tok in must_have)

class OpenAIReflector:
    def __init__(self, model_name: str = "gpt-4o-mini", timeout_seconds: float = 60.0, max_retries: int = 4):
        if OpenAI is None:
            raise RuntimeError("openai package not available. `pip install openai>=1.30`")
        self.client = OpenAI(timeout=timeout_seconds)
        self.model_name = model_name
        self.max_retries = max_retries

    def propose_new_prompt(self, current_instruction: str, casebook: List[Dict[str, Any]]) -> Optional[str]:
        prompt_text = _render_reflection_prompt(current_instruction, casebook)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=800
                )
                text = resp.choices[0].message.content or ""
                new_inst = _parse_backticked_instructions(text)
                if new_inst and _passes_contract(new_inst):
                    return new_inst
                last_err = RuntimeError("Reflector returned invalid or missing backticked instruction.")
            except Exception as e:
                last_err = e
            time.sleep(1.2 * (2 ** attempt))
        return None


# ======================================
# Section 7 — eval helpers (dev accuracy)
# ======================================

def evaluate_split_accuracy(
    system_prompt: str,
    examples: List[TrainExample],
    task_model: SimpleHFChat,
    minibatch_size: int = 8
) -> float:
    total = len(examples)
    if total == 0:
        return 0.0
    correct = 0.0
    for b in range(ceil(total / minibatch_size)):
        start = b * minibatch_size
        end = min((b + 1) * minibatch_size, total)
        _, scores, _ = evaluate_minibatch_once(
            current_system_prompt=system_prompt,
            minibatch_examples=examples[start:end],
            task_model=task_model,
            generation_temperature=0.4,
            generation_top_p=0.95,
            generation_max_new_tokens=192
        )
        correct += sum(scores)
    return correct / total


# ======================================
# Section 8 — acceptance loop (the heart)
# ======================================

def _prompt_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def optimize_instruction_with_reflection(
    seed_system_prompt: str,
    train_examples: List[TrainExample],
    dev_examples: List[TrainExample],
    task_model: SimpleHFChat,
    reflector: OpenAIReflector,
    num_iterations: int = 100,
    minibatch_size: int = 12,
    dev_every: int = 5,
    patience: int = 25,
    global_seed: int = 17
) -> Tuple[str, float]:
    set_all_seeds(global_seed)
    sampler = ClassBalancedSampler(build_label_index(train_examples), global_seed)

    current_prompt = seed_system_prompt
    best_prompt = seed_system_prompt
    best_dev_acc = 0.0
    iters_since_accept = 0

    for iteration in range(1, num_iterations + 1):
        batch_indices = sampler.sample_indices(minibatch_size)
        minibatch = [train_examples[i] for i in batch_indices]

        # evaluate current prompt
        _, old_scores, old_traj = evaluate_minibatch_once(
            current_system_prompt=current_prompt,
            minibatch_examples=minibatch,
            task_model=task_model,
            generation_temperature=0.4,
            generation_top_p=0.95,
            generation_max_new_tokens=192
        )
        old_sum = sum(old_scores)

        if old_sum == len(minibatch):
            print(f"[iter {iteration:03d}] batch perfect ({old_sum}/{len(minibatch)}). skip reflection.")
            iters_since_accept += 1
        else:
            # build casebook
            casebook = build_reflection_casebook(old_traj, k_fail=6, k_pass=2)
            # reflect
            proposed_prompt = reflector.propose_new_prompt(current_prompt, casebook)
            if proposed_prompt is None:
                print(f"[iter {iteration:03d}] reflection failed; keep current.")
                iters_since_accept += 1
            else:
                # strict A/B on same batch
                _, new_scores, _ = evaluate_minibatch_once(
                    current_system_prompt=proposed_prompt,
                    minibatch_examples=minibatch,
                    task_model=task_model,
                    generation_temperature=0.4,
                    generation_top_p=0.95,
                    generation_max_new_tokens=192
                )
                new_sum = sum(new_scores)
                accepted = new_sum > old_sum
                status = "ACCEPT" if accepted else "reject"
                print(f"[iter {iteration:03d}] {status} old={old_sum:.0f} new={new_sum:.0f} prompt={_prompt_sha1(proposed_prompt)}")
                if accepted:
                    current_prompt = proposed_prompt
                    iters_since_accept = 0
                else:
                    iters_since_accept += 1

        if (iteration % dev_every) == 0:
            dev_acc = evaluate_split_accuracy(current_prompt, dev_examples, task_model, minibatch_size=12)
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_prompt = current_prompt
            print(f"[iter {iteration:03d}] DEV acc={dev_acc:.3f} best={best_dev_acc:.3f} prompt={_prompt_sha1(current_prompt)}")

        if iters_since_accept >= patience:
            print(f"[iter {iteration:03d}] stop early; no acceptance for {patience} iters.")
            break

    return best_prompt, best_dev_acc


# =========================
# Section 9 — main script
# =========================

DEFAULT_SEED_PROMPT = (
    "You evaluate a tutor reply for appropriateness of guidance.\n"
    "Think briefly, then answer with EXACTLY one label.\n\n"
    "FORMAT (strict):\n"
    "<think> 2–4 short lines of reasoning </think>\n"
    "<answer> Yes|No|To some extent </answer>\n\n"
    "Rubric:\n"
    "1) Identify the student's specific mistake before giving advice.\n"
    "2) When appropriate, provide at least one concrete next step.\n"
    "3) Avoid generic praise and do not hallucinate details."
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to data/trainset.json")
    parser.add_argument("--target_label", type=str, default="Providing_Guidance", choices=["Providing_Guidance", "Mistake_Identification"])
    parser.add_argument("--hf_task_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--openai_reflector_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--minibatch_size", type=int, default=12)
    parser.add_argument("--dev_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # load + flatten
    raw = load_raw_conversations(args.json_path)
    examples, counts = flatten_to_examples(raw, target_annotation_field=args.target_label)
    print(f"Loaded {len(examples)} examples; label_counts={counts}")

    # split
    train, dev, test = stratified_split(examples, 0.8, 0.1, 0.1, rng_seed=args.seed)
    print(f"Split sizes: train={len(train)} dev={len(dev)} test={len(test)}")

    # task model (HF)
    print(f"Loading HF task model: {args.hf_task_model}")
    task_model = SimpleHFChat(args.hf_task_model)

    # reflector (OpenAI)
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        print("WARNING: OPENAI_API_KEY not set; reflection will fail. Set env var to enable reflector.", file=sys.stderr)
    reflector = OpenAIReflector(model_name=args.openai_reflector_model)

    # optimize
    best_prompt, best_dev_acc = optimize_instruction_with_reflection(
        seed_system_prompt=DEFAULT_SEED_PROMPT,
        train_examples=train,
        dev_examples=dev,
        task_model=task_model,
        reflector=reflector,
        num_iterations=args.iterations,
        minibatch_size=args.minibatch_size,
        dev_every=args.dev_every,
        patience=args.patience,
        global_seed=args.seed
    )

    # final eval on test with best prompt
    test_acc = evaluate_split_accuracy(best_prompt, test, task_model, minibatch_size=12)

    # print & save
    print("\n=== BEST PROMPT ===\n")
    print(best_prompt)
    print(f"\nBEST DEV ACC: {best_dev_acc:.3f} | TEST ACC: {test_acc:.3f}")

    with open("best_prompt.txt", "w", encoding="utf-8") as f:
        f.write(best_prompt)
    with open("final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_dev_acc": best_dev_acc, "test_acc": test_acc}, f, indent=2)
    print("\nSaved best_prompt.txt and final_metrics.json")

if __name__ == "__main__":
    main()
