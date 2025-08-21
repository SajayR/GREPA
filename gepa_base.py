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

"""

from __future__ import annotations
import argparse
import hashlib
import json
import torch
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
import tqdm

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

# =========================
# Track config + seed prompt
# =========================

TRACK_MI = "Mistake Identification"
TRACK_PG = "Providing Guidance"

def get_track_name_from_target(target_field: str) -> str:
    # map your CLI target field to a human-readable track name
    if target_field.strip().lower() in {"mistake_identification", "mistake identification"}:
        return TRACK_MI
    if target_field.strip().lower() in {"providing_guidance", "providing guidance"}:
        return TRACK_PG
    # fallback
    return TRACK_PG

def _shared_output_contract() -> str:
    return (
        "OUTPUT FORMAT (must be exact):\n"
       # "<think>\n"
       # "- Brief internal reasoning (2–5 bullet points).\n"
       # "- End with: Decision = {Yes|No|To some extent}\n"
        #"</think>\n"
        "<answer> Yes|No|To some extent </answer>\n\n"
        "Rules:\n"
        "- Use exactly one of the three labels.\n"
        "- Do not add extra text after </answer>.\n"
        "- Be specific to the student’s content; do not invent facts.\n"
    )

def _track_block(track: str) -> str:
    if track == TRACK_MI:
        return (
            "Track = Mistake Identification\n"
            "Question: Has the tutor identified/recognized a mistake in the student's response?\n\n"
            "Label criteria:\n"
            "- Yes: Precisely names the student’s error (what/where/why). High specificity.\n"
            "- To some extent: Gestures at the error but is vague/incomplete/partly wrong.\n"
            "- No: Fails to identify the mistake or misidentifies it.\n\n"
            "Checklist for <think>:\n"
            "- Quote or paraphrase the exact student bit that’s wrong (one clause).\n"
            "- Name the concept/rule violated.\n"
            "- State in one clause why it’s wrong.\n"
            "- Map: precise → Yes; vague/partial → To some extent; absent/wrong → No.\n"
        )
    else:
        return (
            "Track = Providing Guidance\n"
            "Question: Does the tutor offer correct and relevant guidance (explanation, hint, steps, or example)?\n\n"
            "Label criteria:\n"
            "- Yes: Clear, accurate, task-relevant guidance with at least one concrete next step.\n"
            "- To some extent: Guidance exists but is vague/incomplete/partly inaccurate.\n"
            "- No: No useful guidance, or misleading/incorrect guidance.\n\n"
            "Checklist for <think>:\n"
            "- Is the guidance aligned to the student’s issue?\n"
            "- Is there a concrete next step (e.g., compute/apply/substitute/rewrite/check)?\n"
            "- Is the advice accurate and non-hallucinatory?\n"
            "- Map: clear+accurate+concrete → Yes; vague/partial → To some extent; none/misleading → No.\n"
        )

def build_seed_prompt(track: str) -> str:
    return (
        "You are a strict rater for one task track.\n\n"
        + _shared_output_contract()
        + _track_block(track)
    )



# =================================
# Section 1 — data loading/flatten
# =================================

def load_raw_conversations(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Top-level JSON must be a list"
    return data

def render_input_text(conversation_history: str, candidate_tutor_response: str, track: str = TRACK_PG) -> str:
    if track == TRACK_MI:
        header = "You are evaluating whether the tutor reply identifies the student's mistake according to the rubric."
    else:
        header = "You are evaluating whether the tutor reply provides correct and relevant guidance according to the rubric."
    return (
        f"{header}\n\n"
        "=== DIALOGUE CONTEXT ===\n"
        f"{conversation_history.strip()}\n\n"
        "=== CANDIDATE TUTOR RESPONSE TO EVALUATE ===\n"
        f"{candidate_tutor_response.strip()}\n"
    )


def flatten_to_examples(
    raw_conversations: List[Dict[str, Any]],
    target_annotation_field: str = "Providing_Guidance",  # or "Mistake_Identification"
    track: str = TRACK_PG
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
            input_text = render_input_text(conv_history, resp_text, track)
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
    Minimal HF text-generation wrapper. 
    """
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, dtype: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")
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
        with torch.no_grad():
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

class QwenHFChat:
    """
    Qwen-flavored chat wrapper that:
      - builds proper chat messages (system + user)
      - uses apply_chat_template(..., enable_thinking=True)
      - splits the generated text at the last </think> token (if present)
    Returns both full text (thinking + content) and assistant-only content.
    """
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, dtype: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.model_name = model_name_or_path
        print("Model name loaded")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left"  # important for decoder-only
        )
        print("Tokenizer loaded")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            #device_map=torch.device("cuda"),
            torch_dtype=torch.bfloat16, #dtype if dtype != "auto" else None,
            #trust_remote_code=True
        )
        self.model.to(torch.device("cuda"))
        self.model.eval()
        print("Model loaded internally")

        # Resolve the special token id for </think> once, safely.
        # Do NOT hardcode magic ids like 151668.
        try:
            self.end_think_id = self.tokenizer.convert_tokens_to_ids("</think>")
            if not isinstance(self.end_think_id, int) or self.end_think_id <= 0:
                self.end_think_id = None
        except Exception:
            self.end_think_id = None

    def _render_batch_prompts(self, system_prompt: str, user_texts: List[str]) -> List[str]:
        rendered: List[str] = []
        for user_text in user_texts:
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_text.strip()},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True  # let Qwen open a <think> ... </think> block
            )
            rendered.append(text)
        return rendered

    def generate_full_and_assistant(
        self,
        system_prompt: str,
        user_texts: List[str],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.05,
    ) -> List[tuple[str, str]]:
        """
        Returns a list of (full_text, assistant_text) for each input:
          - full_text: decoded "thinking + content" (nice for logs)
          - assistant_text: only the post-</think> segment (parse this!)
        """
        import torch
        from transformers import GenerationConfig

        prompts = self._render_batch_prompts(system_prompt, user_texts)
        #print("Prompts rendered")
        #print(prompts)
        #print("\n\n\n\n\n\n")
        #print("Prompts: ")
        #for i in range(len(prompts)):
        #    print(f"Prompt {i}: ", prompts[i])
        #    print("\n\n\n\n\n\n")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

        gen_cfg = GenerationConfig(
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)

        results: List[tuple[str, str]] = []
        input_lens = [int(row.shape[0]) for row in inputs.input_ids]  # per-example prompt length

        for i in range(outputs.shape[0]):
            full_ids = outputs[i].tolist()
            tail_ids = full_ids[input_lens[i]:]  # only the generated continuation

            # Split tail at the last </think>, if that token id exists
            assistant_start = 0
            if self.end_think_id is not None:
                try:
                    # reverse search for the last occurrence
                    rev_index = tail_ids[::-1].index(self.end_think_id)
                    last_pos = len(tail_ids) - 1 - rev_index
                    assistant_start = last_pos + 1
                except ValueError:
                    assistant_start = 0  # no </think> found in the tail

            think_ids = tail_ids[:assistant_start]
            content_ids = tail_ids[assistant_start:]
            #print("Think ids: ", think_ids)
            #print("Content ids: ", content_ids)
            thinking_content = self.tokenizer.decode(think_ids, skip_special_tokens=True).strip()
            assistant_content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()
            #print(assistant_content)
            #for i in range(len(thinking_content)):
            #    print(f"Thinking content {i}: ", thinking_content[i])
            #for i in range(len(assistant_content)):
                #print(f"Assistant content {i}: ", assistant_content[i])
            #print("Assistant content: ", assistant_content)
            if thinking_content and assistant_content:
                full_text = (thinking_content + "\n" + assistant_content).strip()
            else:
                # if we couldn't split, the "assistant_content" is the whole decode
                full_text = (thinking_content or assistant_content).strip()

            results.append((full_text, assistant_content))
        #print(results)

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
    task_model: Any, 
    generation_temperature: float = 0.2,
    generation_top_p: float = 0.95,
    generation_max_new_tokens: int = 2048,
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    user_inputs = [ex.input_text for ex in minibatch_examples]
    has_qwen_api = hasattr(task_model, "generate_full_and_assistant")

    try:
        if has_qwen_api:
            pairs = task_model.generate_full_and_assistant(
                system_prompt=current_system_prompt,
                user_texts=user_inputs,
                temperature=generation_temperature,
                top_p=generation_top_p,
                max_new_tokens=generation_max_new_tokens,
            )
            raw_outputs = [ft for (ft, _) in pairs]
            assistant_texts = [at for (_, at) in pairs]
        else:
            raw_outputs = task_model.generate(
                system_prompt=current_system_prompt,
                user_texts=user_inputs,
                temperature=generation_temperature,
                top_p=generation_top_p,
                max_new_tokens=generation_max_new_tokens,
            )
            assistant_texts = list(raw_outputs)
    except Exception:
        raw_outputs = [""] * len(user_inputs)
        assistant_texts = [""] * len(user_inputs)

    scores: List[float] = []
    trajectories: List[Dict[str, Any]] = []
    for ex, full_text, assistant_text in zip(minibatch_examples, raw_outputs, assistant_texts):
        text_for_tag = assistant_text or full_text
        had_tag = ("<answer>" in text_for_tag.lower() and "</answer>" in text_for_tag.lower())
        parsed = parse_answer_tag(assistant_text or full_text)
        off_vocab = bool(had_tag and (parsed is None))  # tag present but not one of allowed labels
        is_correct = 1.0 if (parsed == ex.gold_label) else 0.0

        scores.append(is_correct)
        trajectories.append({
            "input_text": ex.input_text,
            "gold_label": ex.gold_label,
            "model_output_full_text": full_text,
            "assistant_text": assistant_text,
            "parsed_label": parsed,
            "had_answer_tag": had_tag,
            "off_vocab_label": off_vocab,
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

_ACTION_VERBS = {"compute", "calculate", "substitute", "apply", "rewrite",
                 "differentiate", "integrate", "expand", "factor", "check",
                 "verify", "simplify", "try", "show", "solve"}

def _has_action_step(text: str) -> bool:
    lo = (text or "").lower()
    return any(v in lo for v in _ACTION_VERBS)

def _has_pinpoint(text: str) -> bool:
    lo = (text or "").lower()
    # very light signals the model actually points to a concrete error
    return any(p in lo for p in ["mistake", "error", "because", "at step", "incorrect", "should be"]) or ("“" in lo or "\"" in lo)

def short_feedback(track: str, gold: str, pred: Optional[str], had_answer_tag: bool, assistant_text: str) -> str:
    if not had_answer_tag:
        return "Output is missing a single <answer> tag with exactly one of {Yes, No, To some extent}. Add it and remove any extra text after the tag."
    if pred is None:
        return "The label inside <answer> must be exactly one of {Yes, No, To some extent}; do not use synonyms or add words."

    if track == TRACK_MI:
        if gold == "Yes" and pred != "Yes":
            return "Name the specific student error (what/where/why) and reference the offending step; vague topic-level comments are insufficient. The correct answer is Yes."
        if gold == "To some extent" and pred == "No":
            return "You noticed the area but missed precision; state the exact incorrect step or concept and why it’s wrong. The correct answer is To some extent."
        if gold == "No" and pred != "No":
            return "You asserted an error that isn’t present; avoid inventing mistakes and only label Yes when a concrete error is identified. The correct answer is No."
        elif gold != pred:
            return "The correct answer is " + gold + "."
        #if pred == "Yes" and not _has_pinpoint(assistant_text):
        #    return "“Yes” requires precision: point to the exact student step and the violated rule before labeling."
        return "Correct under the MI rubric."
    else:
        # TRACK_PG
        if gold == "Yes" and pred != "Yes":
            return "Provide one concrete, accurate next step aligned with the student’s issue (e.g., compute/apply/substitute/rewrite)."
        if gold == "To some extent" and pred == "No":
            return "Some guidance exists but is vague; add a specific action or example to make it usable."
        if gold == "No" and pred != "No":
            return "The advice is incorrect or off-target; remove misleading steps and only guide when aligned to the student’s need."
        #if pred == "Yes" and not _has_action_step(assistant_text):
        #    return "“Yes” requires at least one actionable step; add a concrete instruction rather than generic encouragement."
        if gold != pred:
            return "The correct answer is " + gold + "."
        return "Correct under the PG rubric."

def build_reflection_casebook(
    trajectories: List[Dict[str, Any]],
    track: str,
    k_fail: int = 6,
    k_pass: int = 2
) -> List[Dict[str, Any]]:
    fails: List[Dict[str, Any]] = []
    passes: List[Dict[str, Any]] = []
    for t in trajectories:
        fb = short_feedback(
            track=track,
            gold=t["gold_label"],
            pred=t.get("parsed_label"),
            had_answer_tag=bool(t.get("had_answer_tag")),
            assistant_text=t.get("assistant_text") or t.get("model_output_full_text") or "",
        )
        rec = {
            "Inputs": t["input_text"],
            "Generated Outputs": t["model_output_full_text"],
            "Feedback": fb,
        }
        if t.get("parsed_label") == t["gold_label"] and t.get("parsed_label") is not None:
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


ALLOWED_LABELS_TEXT = "Yes|No|To some extent"

def _render_reflection_prompt(current_instruction: str, casebook: List[Dict[str, Any]], track: str) -> str:
    def render_item(i: int, rec: Dict[str, Any]) -> str:
        return (
            f"# Example {i}\n"
            f"## Inputs\n{rec['Inputs']}\n\n"
            f"## Model Output\n{rec['Generated Outputs']}\n\n"
            f"## Feedback\n{rec['Feedback']}\n"
        )
    examples_text = "\n\n".join(render_item(i+1, rec) for i, rec in enumerate(casebook))
    return (
        "You are revising an instruction for a classifier LLM that must output EXACTLY one label inside an <answer> tag.\n"
        f"Allowed labels are: {ALLOWED_LABELS_TEXT}\n\n"
        + _shared_output_contract()
        + _track_block(track)
        + "\nHere is the current instruction:\n```"
        f"{current_instruction}"
        "```\n\n"
        "Below are inputs, model outputs, and targeted feedback:\n"
        "```\n"
        f"{examples_text}\n"
        "```\n\n"
        "Analyse the model's outputs and the thinking process--checking for alignment with the feedback and analysing failure cases.\n"
        "Write a revised instruction that:\n"
        "  1) Preserves the exact output format and this track’s label criteria.\n"
        "  2) Strengthens any weak points identified in feedback (precision, concrete steps, alignment, non-hallucination).\n"
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

def count_contract_violations(trajs: List[Dict[str, Any]]) -> Dict[str, int]:
    missing = sum(0 if t.get("had_answer_tag") else 1 for t in trajs)
    offvocab = sum(1 if t.get("off_vocab_label") else 0 for t in trajs)
    return {"missing_tag": missing, "off_vocab": offvocab}

from groq import Groq
from google import genai
from openai import OpenAI
class OpenAIReflector:
    def __init__(self, model_name: str = "deepseek-r1-distill-llama-70b", timeout_seconds: float = 60.0, max_retries: int = 4):
        #if OpenAI is None:
         #   raise RuntimeError("openai package not available. `pip install openai>=1.30`")
        
        #self.client = Groq()
        #self.client = genai.Client()
        self.client = OpenAI()
        self.model_name = model_name
        self.max_retries = max_retries

    def propose_new_prompt(self, current_instruction: str, casebook: List[Dict[str, Any]], track: str) -> Optional[str]:
        prompt_text = _render_reflection_prompt(current_instruction, casebook, track)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                # Save prompt to tmp.txt for debugging
                with open("tmp.txt", "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    #temperature=0.8,
                    #top_p=0.95,
                    #max_tokens=8192
                )
                '''
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_text,
                )'''

                text = resp.choices[0].message.content
                #text = response.text
                print("Text: ", text)
                new_inst = _parse_backticked_instructions(text)
                print("New instruction: ", new_inst)
                if new_inst and _passes_contract(new_inst):
                    return new_inst
                last_err = RuntimeError("Reflector returned invalid or missing backticked instruction.")
            except Exception as e:
                last_err = e
                print("Error: ", e)
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
            generation_max_new_tokens=2048
        )
        correct += sum(scores)
    return correct / total

# ======================================
# Section 7.1 — full-set evaluation utils
# ======================================
def run_full_eval(
    system_prompt: str,
    examples: List[TrainExample],
    task_model: SimpleHFChat,
    batch_size: int = 16,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Returns:
      acc: accuracy on the split
      details: per-example dicts {gold, parsed, ok, output}
    """
    total = len(examples)
    if total == 0:
        return 0.0, []

    correct = 0.0
    details: List[Dict[str, Any]] = []

    for start in tqdm.tqdm(range(0, total, batch_size), desc="Evaluating"):
        ex_batch = examples[start:start + batch_size]
        _, scores, traj = evaluate_minibatch_once(
            current_system_prompt=system_prompt,
            minibatch_examples=ex_batch,
            task_model=task_model,
            generation_temperature=0.4,
            generation_top_p=0.95,
            generation_max_new_tokens=2048
        )
        correct += sum(scores)
        for t in traj:
            details.append({
                "gold": t["gold_label"],
                "parsed": t.get("parsed_label"),
                "ok": int(t.get("parsed_label") == t["gold_label"]),
                "output": t["model_output_full_text"],
            })
    return correct / total, details


def evaluate_prompt_on_splits(
    system_prompt: str,
    splits: Dict[str, List[TrainExample]],
    task_model: SimpleHFChat,
    eval_splits: List[str],
    batch_size: int = 16,
    dump_preds: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate `system_prompt` on chosen splits (e.g., ["dev","test"]).
    If dump_preds is provided, saves a JSONL with one record per example:
      {"split": "...", "gold": "...", "parsed": "...", "ok": 0|1, "output": "..."}
    """
    metrics: Dict[str, float] = {}
    f_out = None
    try:
        if dump_preds:
            f_out = open(dump_preds, "w", encoding="utf-8")

        for name in eval_splits:
            if name not in splits:
                print(f"[WARN] Unknown split '{name}' (available: {list(splits.keys())})")
                continue
            acc, details = run_full_eval(system_prompt, splits[name], task_model, batch_size=batch_size)
            metrics[name] = acc
            print(f"[EVAL] {name.upper()} n={len(splits[name])} acc={acc:.3f}")

            if f_out:
                for d in details:
                    rec = {"split": name, **d}
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if f_out:
            f_out.close()
            print(f"[EVAL] wrote predictions to {dump_preds}")
    return metrics



# ======================================
# Section 8 — acceptance loop (the heart)
# ======================================

def _prompt_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def optimize_instruction_with_reflection(
    seed_system_prompt: str,
    train_examples: List[TrainExample],
    dev_examples: List[TrainExample],
    task_model: Any,
    reflector: OpenAIReflector,
    num_iterations: int = 100,
    minibatch_size: int = 12,
    dev_every: int = 5,
    patience: int = 25,
    global_seed: int = 17,
    track: str = TRACK_PG,
) -> Tuple[str, float]:
    set_all_seeds(global_seed)
    sampler = ClassBalancedSampler(build_label_index(train_examples), global_seed)

    current_prompt = seed_system_prompt
    best_prompt = seed_system_prompt
    best_dev_acc = 0.0
    iters_since_accept = 0

    for iteration in tqdm.tqdm(range(1, num_iterations + 1)):
        batch_indices = sampler.sample_indices(minibatch_size)
        minibatch = [train_examples[i] for i in batch_indices]

        # evaluate current prompt
        _, old_scores, old_traj = evaluate_minibatch_once(
            current_system_prompt=current_prompt,
            minibatch_examples=minibatch,
            task_model=task_model,
            generation_temperature=0.4,
            generation_top_p=0.95,
            generation_max_new_tokens=2048
        )
        print("Old scores: ", old_scores)
        old_sum = sum(old_scores)
        old_v = count_contract_violations(old_traj)

        if old_sum == len(minibatch):
            print(f"[iter {iteration:03d}] batch perfect ({old_sum}/{len(minibatch)}). skip reflection.")
            iters_since_accept += 1
        else:
            casebook = build_reflection_casebook(old_traj, track=track, k_fail=6, k_pass=2)
            proposed_prompt = reflector.propose_new_prompt(current_prompt, casebook, track=track)
            if proposed_prompt is None:
                print(f"[iter {iteration:03d}] reflection failed; keep current.")
                iters_since_accept += 1
            else:
                # strict A/B on same batch
                _, new_scores, new_traj = evaluate_minibatch_once(
                    current_system_prompt=proposed_prompt,
                    minibatch_examples=minibatch,
                    task_model=task_model,
                    generation_temperature=0.4,
                    generation_top_p=0.95,
                    generation_max_new_tokens=2048
                )
                new_sum = sum(new_scores)
                new_v = count_contract_violations(new_traj)

                # Acceptance rule:
                # 1) must improve accuracy; and
                # 2) must not increase contract violations; and
                # 3) off-vocab must be zero (hard guard).
                improved = new_sum > old_sum
                contract_ok = (new_v["missing_tag"] <= old_v["missing_tag"]) and (new_v["off_vocab"] <= old_v["off_vocab"])
                hard_guard = (new_v["off_vocab"] == 0)

                accepted = bool(improved and contract_ok and hard_guard)
                status = "ACCEPT" if accepted else "reject"
                if accepted:
                    with open("accepted_prompt.txt", "w", encoding="utf-8") as f:
                        f.write(proposed_prompt)
                    
                more = f"old={old_sum:.0f}/{len(minibatch)} v={old_v} new={new_sum:.0f}/{len(minibatch)} v={new_v}"
                print(f"[iter {iteration:03d}] {status} {more} prompt={_prompt_sha1(proposed_prompt)}")

                if accepted:
                    current_prompt = proposed_prompt
                    iters_since_accept = 0
                else:
                    # secondary tie-break: if accuracy ties but strictly fewer violations, you may allow:
                    if (new_sum == old_sum) and ( (new_v["missing_tag"] < old_v["missing_tag"]) or (new_v["off_vocab"] < old_v["off_vocab"]) ) and hard_guard:
                        print(f"[iter {iteration:03d}] ACCEPT (tie on acc, fewer violations).")
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
    #"<think> 2–4 short lines of reasoning </think>\n"
    "<answer> Yes|No|To some extent </answer>\n\n"
    "Rubric:\n"
    "1) Identify the student's specific mistake before giving advice.\n"
    "2) When appropriate, provide at least one concrete next step.\n"
    "3) Avoid generic praise and do not hallucinate details.\n"
    "4) IT IS CRITICAL THAT YOU ENCLOSE THE ANSWER IN <answer> TAGS, OTHERWISE YOUR ANSWER WILL BE IGNORED\n"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["optimize", "evaluate"], default="optimize",
                        help="optimize: run reflective search; evaluate: score a given prompt.")
    parser.add_argument("--json_path", type=str, default="/speedy/CisStuff/IndoML/data/trainset.json", help="Path to data/trainset.json")
    parser.add_argument("--target_label", type=str, default="Mistake_Identification",
                        choices=["Providing_Guidance", "Mistake_Identification"])
    parser.add_argument("--hf_task_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--openai_reflector_model", type=str, default="deepseek-r1-distill-llama-70b")

    # optimize-mode knobs
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--minibatch_size", type=int, default=4)
    parser.add_argument("--dev_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

    # evaluate-mode knobs
    parser.add_argument("--prompt_text", type=str, default=None, help="Inline prompt text to evaluate.")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to a text file with the prompt.")
    parser.add_argument("--eval_splits", type=str, default="dev,test", help="Comma list among train,dev,test.")
    parser.add_argument("--dump_preds", type=str, default=None, help="Path to write JSONL predictions.")

    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # load + flatten
    track = get_track_name_from_target(args.target_label)
    raw = load_raw_conversations(args.json_path)
    examples, counts = flatten_to_examples(raw, target_annotation_field=args.target_label, track=track)
    print(f"Loaded {len(examples)} examples; label_counts={counts}")

    # split
    train, dev, test = stratified_split(examples, 0.8, 0.1, 0.1, rng_seed=args.seed)
    print(f"Split sizes: train={len(train)} dev={len(dev)} test={len(test)}")

    
    seed_prompt = build_seed_prompt(track)


    # task model (HF)
    print(f"Loading HF task model: {args.hf_task_model}")
    #task_model = SimpleHFChat(args.hf_task_model)
    task_model = QwenHFChat(args.hf_task_model)
    print("Model loaded")
    if args.mode == "evaluate":
        # decide prompt
        prompt = None
        if args.prompt_text:
            prompt = args.prompt_text
        elif args.prompt_path and os.path.exists(args.prompt_path):
            prompt = open(args.prompt_path, "r", encoding="utf-8").read()
        else:
            prompt = seed_prompt
            print("[EVAL] No --prompt_text/--prompt_path provided; using track-specific seed prompt.")

        # which splits to eval
        wanted = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
        splits = {"train": train, "dev": dev, "test": test}

        # run evaluation
        evaluate_prompt_on_splits(
            system_prompt=prompt,
            splits=splits,
            task_model=task_model,
            eval_splits=wanted,
            batch_size=args.minibatch_size,
            dump_preds=args.dump_preds
        )
        return

    # === optimize mode (default) ===
    # reflector (OpenAI)
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set; reflection will fail. Set env var to enable reflector.", file=sys.stderr)
    reflector = OpenAIReflector(model_name=args.openai_reflector_model)

    # optimize
    best_prompt, best_dev_acc = optimize_instruction_with_reflection(
        seed_system_prompt=seed_prompt,
        train_examples=train,
        dev_examples=dev,
        task_model=task_model,
        reflector=reflector,
        num_iterations=args.iterations,
        minibatch_size=args.minibatch_size,
        dev_every=args.dev_every,
        patience=args.patience,
        global_seed=args.seed,
        track=track
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