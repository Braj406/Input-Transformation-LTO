"""Dataset loading, prompt construction, and answer parsing for CommonsenseQA / GSM8K."""
import random
import re

import numpy as np
from datasets import load_dataset

_REPO = {"commonsense_qa": ("tau/commonsense_qa", None), "gsm8k": ("openai/gsm8k", "main")}


def load_examples(name, n):
    repo_id, config = _REPO[name]
    split = "validation" if name == "commonsense_qa" else "test"
    ds = load_dataset(repo_id, config, split=split)
    info = {"repo_id": repo_id, "version": str(ds.info.version), "full_rows": ds.num_rows}
    return ds.select(range(min(n, len(ds)))), info


def load_exemplar_pool(name, n_pool=64, seed=0):
    """Style exemplars come from the TRAIN split -- never from the split being scored."""
    repo_id, config = _REPO[name]
    ds = load_dataset(repo_id, config, split="train")
    idxs = random.Random(seed + 777).sample(range(len(ds)), min(n_pool, len(ds)))
    return [ds[i]["question"] for i in idxs]


def cqa_user(q, choices):
    lines = [f"{lab}. {txt}" for lab, txt in zip(choices["label"], choices["text"])]
    return f"Question: {q}\nChoices:\n" + "\n".join(lines) + "\nRespond with the single letter of the correct choice."


def gsm_user(q):
    return f"Question: {q}\nSolve step by step, then give the final numeric answer after '####'."


def build_user_and_gold(name, ex):
    if name == "commonsense_qa":
        return cqa_user(ex["question"], ex["choices"]), ex["answerKey"]
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ex["answer"])
    return gsm_user(ex["question"]), (float(m.group(1).replace(",", "")) if m else None)


_CQA_CUES = [
    r"final\s*answer\s*[:\-]?\s*\(?([A-E])\)?", r"answer\s*is\s*\(?([A-E])\)?",
    r"answer\s*[:\-]\s*\(?([A-E])\)?", r"choose\s*(?:option\s*)?\(?([A-E])\)?",
    r"option\s*\(?([A-E])\)?", r"\(([A-E])\)",
]


def parse_cqa(text):
    for pat in _CQA_CUES:
        m = list(re.finditer(pat, text, re.IGNORECASE))
        if m:
            return m[-1].group(1).upper()
    m = list(re.finditer(r"\b([A-E])\b", text))
    return m[-1].group(1).upper() if m else None


def _num(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except Exception:
        return None


def parse_gsm(text):
    m = list(re.finditer(r"####\s*(-?[\d,]+(?:\.\d+)?)", text))
    if m:
        return _num(m[-1].group(1))
    m = list(re.finditer(r"(?:answer|total|=)\s*(?:is\s*)?\$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE))
    if m:
        return _num(m[-1].group(1))
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return _num(nums[-1]) if nums else None


def parse_pred(name, text):
    return parse_cqa(text) if name == "commonsense_qa" else parse_gsm(text)


def is_correct(name, pred, gold):
    if pred is None or gold is None:
        return False
    return pred == gold if name == "commonsense_qa" else abs(pred - gold) < 1e-4


def earliest_correct_step(name, gen_ids, tok, gold, stride=1):
    first, flags, last = -1, [], False
    for gi in range(len(gen_ids)):
        if stride > 1 and gi % stride and gi != len(gen_ids) - 1:
            flags.append(last)
            continue
        last = is_correct(name, parse_pred(name, tok.decode(gen_ids[:gi + 1], skip_special_tokens=True)), gold)
        flags.append(last)
        if last and first == -1:
            first = gi
    return first, np.asarray(flags, dtype=bool)
