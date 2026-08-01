"""Deterministic (greedy) generation with per-step hidden-state / metric extraction."""
import numpy as np
import pandas as pd
import torch

from .metrics import (
    METRIC_NAMES,
    calculate_anisotropy,
    calculate_effective_rank,
    calculate_intrinsic_dimension,
    compute_matrix_entropy,
)


@torch.inference_mode()
def extract_compact(tok, model, prompt_text, cfg, stop_ids):
    """
    ANSWER INFERENCE -- DETERMINISTIC. Pure argmax == temperature 0, no sampling, no top_p.
    Same prompt always yields the same answer and the same hidden states.
    """
    assert cfg.get("inference_greedy", True), "inference must be greedy"
    device = model.device
    L = model.config.num_hidden_layers
    enc = tok(prompt_text, return_tensors="pt").to(device)
    attn = enc.attention_mask
    out = model(input_ids=enc.input_ids, attention_mask=attn, use_cache=True, output_hidden_states=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    del out
    accum, gen_ids = None, []
    for _ in range(cfg["max_new_tokens"]):
        nid = logits.argmax(-1)
        tid = int(nid)          # <-- temperature 0
        if tid in stop_ids:
            break
        gen_ids.append(tid)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], 1)
        out = model(input_ids=nid.view(1, 1), attention_mask=attn, past_key_values=past,
                    use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = torch.stack([out.hidden_states[l][0, -1, :] for l in range(L + 1)], 0).float().unsqueeze(1)
        accum = h if accum is None else torch.cat([accum, h], 1)
        logits = out.logits[:, -1, :]
        del out
    T = len(gen_ids)
    if T == 0:
        hidden, metrics = torch.empty(0), torch.empty(0)
    else:
        rows = []
        for l in range(L + 1):
            M = accum[l]
            if M.shape[0] > cfg["max_metric_tokens"]:
                M = M[torch.randperm(M.shape[0], device=M.device)[:cfg["max_metric_tokens"]]]
            rows.append([compute_matrix_entropy(M), calculate_effective_rank(M),
                         calculate_anisotropy(M), calculate_intrinsic_dimension(M)])
        metrics = torch.tensor(rows, dtype=torch.float32)
        if cfg["hidden_pooling"] == "mean":
            hidden = accum.mean(1).to(torch.float16).cpu()
        elif cfg["hidden_pooling"] == "last":
            hidden = accum[:, -1, :].to(torch.float16).cpu()
        else:
            hidden = accum.permute(1, 0, 2).to(torch.float16).cpu()
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del past, logits, enc
    if accum is not None:
        del accum
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"generated_text": text, "gen_len": T, "gen_ids": gen_ids, "hidden": hidden, "metrics": metrics}


def _mean_over_layers(m):
    if m.numel() == 0:
        return {k: float("nan") for k in METRIC_NAMES}
    v = m.mean(0)
    return {METRIC_NAMES[i]: float(v[i]) for i in range(4)}


def write_multi_csv(records, path):
    rows = []
    for r in records:
        o = r["original"]
        mo = _mean_over_layers(o["metrics"])
        for vkey, v in r["variants"].items():
            for s in v["samples"]:
                mt = _mean_over_layers(s["metrics"])
                rows.append({
                    "idx": r["idx"], "gold": r["gold"], "transform": vkey,
                    "sample": s["sample"], "applied": s["applied"], "duplicate": s.get("duplicate", False),
                    "guard": s.get("guard"), "variation": s.get("variation"),
                    "leak_flag": s.get("leak_flag", False),
                    "orig_pred": o["pred"], "orig_correct": o["correct"],
                    "tf_pred": s["pred"], "tf_correct": s["correct"],
                    "flip_to_correct": s["flip_to_correct"], "flip_to_incorrect": s["flip_to_incorrect"],
                    "gen_len_orig": o["gen_len"], "gen_len_tf": s["gen_len"],
                    "tf_question": s.get("tf_question"),
                    **{f"orig_{k}": mo[k] for k in METRIC_NAMES},
                    **{f"tf_{k}": mt[k] for k in METRIC_NAMES},
                })
    pd.DataFrame(rows).to_csv(path, index=False)


CONF_FEATURES = ["mean_maxp", "min_maxp", "last_maxp", "first_maxp", "mean_entropy", "max_entropy"]


@torch.inference_mode()
def extract_confidence(tok, model, prompt_text, cfg, stop_ids):
    """Second deterministic pass (no hidden states) that recovers per-token confidence:
    the argmax probability and predictive entropy at each generated step."""
    device = model.device
    enc = tok(prompt_text, return_tensors="pt").to(device)
    attn = enc.attention_mask
    out = model(input_ids=enc.input_ids, attention_mask=attn, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    del out
    gen_ids, maxp, ent = [], [], []
    for _ in range(cfg["max_new_tokens"]):
        probs = torch.softmax(logits.float(), dim=-1)
        nid = logits.argmax(-1)
        tid = int(nid)
        if tid in stop_ids:
            break
        maxp.append(probs[0, tid].item())
        ent.append((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
        gen_ids.append(tid)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], 1)
        out = model(input_ids=nid.view(1, 1), attention_mask=attn, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        del out
    text = tok.decode(gen_ids, skip_special_tokens=True)
    del past, logits, enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not maxp:
        return dict(text=text, mean_maxp=float("nan"), min_maxp=float("nan"),
                    last_maxp=float("nan"), first_maxp=float("nan"),
                    mean_entropy=float("nan"), max_entropy=float("nan"))
    return dict(text=text, mean_maxp=float(np.mean(maxp)), min_maxp=float(np.min(maxp)),
                last_maxp=float(maxp[-1]), first_maxp=float(maxp[0]),
                mean_entropy=float(np.mean(ent)), max_entropy=float(np.max(ent)))
