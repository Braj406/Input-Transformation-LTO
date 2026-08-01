"""LLM-driven question rewrites: paraphrase/clarify/simplify styles, in-context-style
rewrites, generated-knowledge append, and a meaning-null format-jitter control.

Rewrites are GREEDY (temperature 0.0) by default, same as answer inference. Diversity
across k comes from PROMPT VARIATION, not from sampling:
  llm_paraphrase -> sample s uses style directive s   (named, rankable)
  llm_icr        -> sample s uses exemplar draw s (from the TRAIN split)
  llm_knowledge  -> generated-knowledge prompting (APPENDS a fact)
  llm_clarify / llm_simplify / llm_simplify_v2 / llm_simplify_v3 -> pinned style, k=1 each
  format_jitter  -> meaning-null surface control (the variance baseline)

NOTE on fixed-style types: their prompt is FIXED (same text every attempt/sample), so at
temperature 0 every attempt is byte-identical. Use k=1 for these -- k>1 just regenerates
duplicates. To get more than one usable rewrite per question for a fixed-style family, add
MORE NAMED VARIANTS (like llm_simplify_v2/v3) rather than raising k. Retries (attempts>1)
DO help though: if attempt 0 fails a guard, later attempts fall back to a small temperature
so they can actually produce something different instead of failing the same guard again.
"""
import hashlib
import random
import re

import torch

from ..data import cqa_user, gsm_user
from .char_word import char_transform, combo_transform, word_transform, _pack_meta

REWRITE_TEMPERATURE = 0.0

_STYLE_DIRECTIVES = [
    "Rewrite it as one clear, direct question in plain everyday language.",
    "Rewrite it so that every implicit assumption is stated explicitly.",
    "Rewrite it using simpler and more common words, keeping it short.",
    "Rewrite it in a more formal and precise register.",
    "Rewrite it by reordering the clauses so the actual question comes first.",
    "Rewrite it as a short scenario followed by the question.",
    "Rewrite it so that a child could understand it, without losing any detail.",
    "Rewrite it as a single sentence with no subordinate clauses.",
]
_FIXED_STYLE = {
    "llm_clarify":     "Rewrite it so that every implicit assumption and every referent is stated explicitly and unambiguously.",
    "llm_simplify":    "Rewrite it using the simplest possible words and the shortest possible sentences.",
    "llm_simplify_v2": "Rewrite it as a short, plain-language version a non-native speaker could easily follow.",
    "llm_simplify_v3": "Rewrite it by removing all stylistic flourishes, keeping only the core facts and the question.",
}

_PARAPHRASE_TMPL = (
    "You rewrite questions for a benchmark. Keep EXACTLY the same meaning and the same correct answer.\n\n"
    "### Style instruction ###\n{style}\n\n"
    "### Example ###\n"
    "Original: Sammy wanted to go to where the people were. Where might he go?\n"
    "Rewritten: Where would Sammy head if he wanted to be around a crowd of people?\n\n"
    "### Rules ###\n"
    "- You MUST change the wording. Never copy the original sentence verbatim.\n"
    "- Keep every number, name, and quantity exactly the same.\n"
    "- Do not add, remove, or change any factual detail.\n"
    "- Do not answer the question or hint at the answer.\n"
    "- Output ONLY the rewritten question.\n\nOriginal: {q}\nRewritten:")

_ICR_TMPL = (
    "You rewrite questions to match the style of a target dataset, keeping EXACTLY the same meaning "
    "and the same correct answer.\n\n"
    "### Target style examples ###\n{examples}\n\n"
    "### Rules ###\n"
    "- You MUST change the wording. Never copy the original sentence verbatim.\n"
    "- Match the phrasing style of the examples above.\n"
    "- Keep every number, name, and quantity exactly the same.\n"
    "- Do not add, remove, or change any factual detail.\n"
    "- Do not answer the question or hint at the answer.\n"
    "- Output ONLY the rewritten question.\n\nOriginal: {q}\nRewritten:")

# Generated-knowledge prompting. The rewriter NEVER sees the choices or the gold answer,
# so any useful fact it produces is the model's own knowledge, not leakage.
_KNOW_TMPL = (
    "Write ONE short sentence of general world knowledge that would help someone answer the "
    "question below.\n\n"
    "### Style examples of the kinds of questions this is for ###\n{examples}\n\n"
    "### Rules ###\n"
    "- Do NOT answer the question.\n"
    "- Do NOT mention or invent any answer options.\n"
    "- State a general fact, not a fact about this specific situation.\n"
    "- Output ONLY the single sentence.\n\nQuestion: {q}\nKnowledge:")

# Meaning-null surface control -- the variance baseline. Any pass@k it earns is pass@k
# that a real transform must beat to mean anything.
_JITTERS = [
    lambda q: "Consider the following. " + q,
    lambda q: '"' + q.strip() + '"',
    lambda q: "Q: " + q,
    lambda q: "[1] " + q,
    lambda q: "Question text: " + q,
    lambda q: q.strip() + " (end of question)",
]

APPEND_TYPES = {"llm_knowledge"}
REWRITE_TYPES = {"llm_paraphrase", "llm_icr", "llm_clarify", "llm_simplify", "llm_simplify_v2", "llm_simplify_v3"}
LLM_TYPES = APPEND_TYPES | REWRITE_TYPES


@torch.inference_mode()
def _llm_generate(model, tok, user_prompt, max_new_tokens=96, temperature=0.0, top_p=1.0):
    """Rewrite generation. Greedy by default -- identical to answer inference."""
    from ..model_io import format_prompt
    p = format_prompt(tok, user_prompt, enable_thinking=False)
    inputs = tok(p, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False)
    out = model.generate(**inputs, **kw)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _numbers(t):
    return sorted(re.findall(r"-?\d+(?:\.\d+)?", t.replace(",", "")))


def _draw_exemplars(pool, n, idx, vkey, variation, seed):
    """Deterministic per-(example, transform, sample, attempt) exemplar draw."""
    h = hashlib.sha1(f"ex|{seed}|{idx}|{vkey}|{variation}".encode()).hexdigest()[:12]
    return random.Random(int(h, 16)).sample(pool, min(n, len(pool)))


def _clean_rewrite(raw, orig=None):
    t = raw.strip()
    m = list(re.finditer(r'(?i)\b(?:rewritten|knowledge)\s*:', t))
    if m:
        t = t[m[-1].end():]
    t = t.strip().split("\n\n")[0].strip()
    t = re.sub(r'^\s*(?:rewritten(?:\s+question)?|paraphrased(?:\s+(?:text|question))?|'
               r'knowledge|original|question|answer)\s*[:\-]\s*', '', t, flags=re.I)
    t = t.strip().strip('`"\'')
    if orig and len(t) > 3 * len(orig) + 80:
        t = t.split("\n")[0].strip()
    return t


def _check_guards(orig, rw):
    """Rewrite mode: meaning must survive exactly."""
    if (not rw) or rw.strip().lower() == orig.strip().lower():
        return False, "empty_or_identical"
    if _numbers(rw) != _numbers(orig):
        return False, "number_mismatch"
    return True, "ok"


def _check_guards_append(orig, add):
    """Append mode: the original must survive; the addition must not answer the question."""
    if not add or len(add) < 10:
        return False, "empty_or_short"
    if len(add) > 3 * len(orig) + 200:
        return False, "too_long"
    if re.search(r"\b(the answer is|answer\s*[:=]|correct (option|choice|answer))\b", add, re.I):
        return False, "answers_question"
    if not set(_numbers(orig)).issubset(set(_numbers(orig + " " + add))):
        return False, "number_mismatch"
    return True, "ok"


def _leakage_check(orig_q, rw_q, ex, dataset_name):
    if dataset_name != "commonsense_qa":
        return True, "ok"
    labels, texts = ex["choices"]["label"], ex["choices"]["text"]
    gold_text = next((t for l, t in zip(labels, texts) if l == ex.get("answerKey")), None)
    if not gold_text:
        return True, "ok"
    g = gold_text.strip().lower()
    if not g:
        return True, "ok"
    pat = r"\b" + re.escape(g) + r"\b"
    if (not re.search(pat, orig_q.lower())) and re.search(pat, rw_q.lower()):
        return False, "answer_leak"
    return True, "ok"


def llm_rewrite(text, tf_cfg, llm_ctx, idx=0, sample=0, vkey="v", seed=0):
    """Deterministic on the first attempt. For variation-indexed types (paraphrase/icr/
    knowledge), each attempt uses a DIFFERENT prompt variation. For FIXED_STYLE types
    (clarify/simplify* -- same prompt every attempt), a repeat at temperature 0 would be
    byte-identical and could never recover from a guard failure, so retries (a>0) fall
    back to a small temperature instead."""
    model, tok = llm_ctx["model"], llm_ctx["tok"]
    pool = llm_ctx["exemplar_pool"]
    ttype = tf_cfg["type"]
    attempts = tf_cfg.get("attempts", 3)
    temp = tf_cfg.get("temperature", REWRITE_TEMPERATURE)
    top_p = tf_cfg.get("top_p", 1.0)
    ntok = tf_cfg.get("rewrite_tokens", 96)
    n_ex = tf_cfg.get("n_exemplars", 4)

    last_reason, variant_id = "empty_or_identical", None
    for a in range(attempts):
        variation = sample * 101 + a          # distinct, deterministic, per (sample, attempt)
        gen_temp = temp if ttype not in _FIXED_STYLE else (0.0 if a == 0 else 0.5)

        if ttype == "llm_paraphrase":
            si = variation % len(_STYLE_DIRECTIVES)
            prompt = _PARAPHRASE_TMPL.replace("{style}", _STYLE_DIRECTIVES[si]).replace("{q}", text)
            vid = f"style{si}"
        elif ttype in _FIXED_STYLE:
            prompt = _PARAPHRASE_TMPL.replace("{style}", _FIXED_STYLE[ttype]).replace("{q}", text)
            vid = "fixed" if a == 0 else f"fixed_retry{a}"
        elif ttype == "llm_icr":
            exs = _draw_exemplars(pool, n_ex, idx, vkey, variation, seed)
            prompt = _ICR_TMPL.replace("{examples}", "\n".join(f"- {e}" for e in exs)).replace("{q}", text)
            vid = f"ex{variation}"
        elif ttype == "llm_knowledge":
            exs = _draw_exemplars(pool, n_ex, idx, vkey, variation, seed)
            prompt = _KNOW_TMPL.replace("{examples}", "\n".join(f"- {e}" for e in exs)).replace("{q}", text)
            vid = f"ex{variation}"
        else:
            return text, {"transform_type": ttype, "num_changed": 0, "guard": "unknown_type",
                          "similarity": None, "attempts_used": 0,
                          "original_text": text, "rewritten_text": text, "variation": None}

        gen = _clean_rewrite(_llm_generate(model, tok, prompt, ntok, gen_temp, top_p), text)

        if ttype in APPEND_TYPES:
            ok, reason = _check_guards_append(text, gen)
            new_text = f"{text}\nRelevant background: {gen}" if ok else text
        else:
            ok, reason = _check_guards(text, gen)
            new_text = gen if ok else text

        last_reason, variant_id = reason, vid
        if ok:
            return new_text, {"transform_type": ttype, "num_changed": 1, "guard": "ok",
                              "similarity": None, "attempts_used": a + 1, "variation": vid,
                              "original_text": text, "rewritten_text": new_text}

    return text, {"transform_type": ttype, "num_changed": 0, "guard": last_reason,
                  "similarity": None, "attempts_used": attempts, "variation": variant_id,
                  "original_text": text, "rewritten_text": text}


def format_jitter(text, tf_cfg, sample=0):
    """Meaning-null surface perturbation. THIS IS THE CONTROL -- any pass@k it earns is
    pass@k that a real transform must beat to mean anything."""
    j = sample % len(_JITTERS)
    new = _JITTERS[j](text)
    ok = new.strip().lower() != text.strip().lower()
    return (new if ok else text), {"transform_type": "format_jitter",
                                   "num_changed": 1 if ok else 0,
                                   "guard": "ok" if ok else "empty_or_identical",
                                   "similarity": None, "variation": f"jit{j}",
                                   "original_text": text, "rewritten_text": new if ok else text}


def transform_text(text, tf_cfg, llm_ctx=None, idx=0, sample=0, vkey="v", seed=0):
    t = tf_cfg["type"]
    if t == "char":
        return char_transform(text, tf_cfg.get("mode", "swap"), tf_cfg.get("p", 1.0), tf_cfg.get("edits", 1))
    if t == "word":
        return word_transform(text, tf_cfg.get("p", 0.5), tf_cfg.get("strategy", "frequency"))
    if t == "combo":
        return combo_transform(text, tf_cfg.get("p_word", 0.5), tf_cfg.get("p_char", 0.5),
                               tf_cfg.get("mode", "swap"), tf_cfg.get("strategy", "frequency"),
                               tf_cfg.get("edits", 1))
    if t == "format_jitter":
        return format_jitter(text, tf_cfg, sample)
    if t in LLM_TYPES:
        assert llm_ctx is not None, "LLM transforms require llm_ctx."
        return llm_rewrite(text, tf_cfg, llm_ctx, idx=idx, sample=sample, vkey=vkey, seed=seed)
    return text, _pack_meta("none", 0.0, [])


def variant_key(tf):
    t = tf["type"]
    if t == "combo":
        return f"combo_pw{tf.get('p_word', 0.5)}_pc{tf.get('p_char', 0.5)}"
    if t == "word":
        return f"synonym_p{tf.get('p', 0.5)}"
    if t == "char":
        return f"{tf.get('mode', 'swap')}_p{tf.get('p', 1.0)}"
    if t == "format_jitter":
        return "ctrl_format_jitter"
    if t in LLM_TYPES:
        return f"{t}_g"          # _g = greedy rewrite
    return "none"


def build_transformed_user(name, ex, tf_cfg, llm_ctx=None, idx=0, sample=0, vkey="v", seed=0):
    q_new, meta = transform_text(ex["question"], tf_cfg, llm_ctx=llm_ctx,
                                 idx=idx, sample=sample, vkey=vkey, seed=seed)
    ttype = tf_cfg["type"]
    default_leak = ttype in REWRITE_TYPES
    if tf_cfg.get("check_leakage", default_leak) and meta.get("num_changed", 0) > 0:
        ok, reason = _leakage_check(ex["question"], q_new, ex, name)
        if not ok:
            q_new = ex["question"]
            meta = {**meta, "num_changed": 0, "guard": reason, "rewritten_text": ex["question"]}
    elif ttype in APPEND_TYPES and meta.get("num_changed", 0) > 0:
        ok, _ = _leakage_check(ex["question"], q_new, ex, name)
        meta["leak_flag"] = (not ok)          # audited, not rejected
    meta["transformed_question"] = q_new
    return (cqa_user(q_new, ex["choices"]) if name == "commonsense_qa" else gsm_user(q_new)), meta
