"""End-to-end runner: generate baseline + transformed trajectories for a batch of
questions, incrementally checkpointing to Google Drive. Exemplar pool for style-matched
rewrites is drawn from the TRAIN split -- no eval contamination.
"""
import datetime
import gc
import json
import os
import shutil
from collections import Counter

import torch
from tqdm.auto import tqdm

from .config import COMPACT_CFG, DEVICE, MODELS, SEED, set_seed
from .data import build_user_and_gold, earliest_correct_step, is_correct, load_examples, load_exemplar_pool, parse_pred
from .extraction import extract_compact, write_multi_csv
from .model_io import format_prompt, load_model, sanity_check, stop_token_ids
from .transforms import LLM_TYPES, build_transformed_user, variant_key, variant_seed


def _vote(preds, fallback):
    """Majority vote over predictions; ties broken toward the original prediction."""
    preds = [p for p in preds if p not in (None, "None", "")]
    if not preds:
        return fallback
    c = Counter(preds)
    top = max(c.values())
    tied = [p for p, n in c.items() if n == top]
    return fallback if fallback in tied else sorted(tied)[0]


def run_multi(model_key, dataset_name, transforms, n_examples, drive_dir, cfg=None, resume=True,
              out_name=None, n_pool=64, seed=SEED):
    cfg = dict(cfg or COMPACT_CFG)
    spec = MODELS[model_key]
    tag = f"{spec['short']}__{dataset_name}" + (f"__{out_name}" if out_name else "")
    drive_pt = os.path.join(drive_dir, tag + ".pt")
    drive_csv = os.path.join(drive_dir, tag + ".csv")
    drive_cfg = os.path.join(drive_dir, tag + ".config.json")
    local_pt = f"/content/{tag}.temp.pt" if os.path.isdir("/content") else drive_pt + ".temp"
    keys = [variant_key(t) for t in transforms]
    ks = [t.get("k", 1) for t in transforms]

    if resume and os.path.exists(drive_pt):
        shutil.copy2(drive_pt, local_pt)
        records = torch.load(local_pt, weights_only=False)
    else:
        records = []
    by_idx = {r["idx"]: r for r in records}
    print(f"{tag}: {len(records)} existing records; transforms = {list(zip(keys, ks))}")

    tok, model, _ = load_model(model_key)
    sanity_check(tok, model, spec)
    stop_ids = stop_token_ids(tok, model)
    examples, ds_info = load_examples(dataset_name, n_examples)
    think = cfg["enable_thinking"] if model_key.startswith("qwen3") else None

    llm_needed = any(t["type"] in LLM_TYPES for t in transforms)
    exemplar_pool = load_exemplar_pool(dataset_name, n_pool, seed=seed) if llm_needed else []
    llm_ctx = {"model": model, "tok": tok, "exemplar_pool": exemplar_pool} if llm_needed else None
    if llm_needed:
        print(f"exemplar pool: {len(exemplar_pool)} questions from TRAIN split")

    os.makedirs(drive_dir, exist_ok=True)
    json.dump({"model": spec["id"], "dataset": dataset_name, "dataset_info": ds_info,
               "transforms": transforms, "variant_keys": keys, "k_per_transform": ks, "seed": seed,
               "inference": "greedy / temperature=0 (deterministic)",
               "rewrite": "greedy / temperature=0; diversity from style index + exemplar draw",
               "exemplar_pool_split": "train", "exemplar_pool_size": len(exemplar_pool),
               "hidden_pooling": cfg["hidden_pooling"], "max_new_tokens": cfg["max_new_tokens"],
               "created": datetime.datetime.now().isoformat()},
              open(drive_cfg, "w"), indent=2, default=str)

    def _flush():
        recs = list(by_idx.values())
        torch.save(recs, local_pt)
        shutil.copy2(local_pt, drive_pt)
        write_multi_csv(recs, drive_csv)

    n_new = 0
    pbar = tqdm(range(len(examples)), desc=tag)
    for idx in pbar:
        rec = by_idx.get(idx)
        need = any((rec is None) or (k not in rec.get("variants", {}))
                   or (len(rec["variants"][k]["samples"]) < kk)
                   for k, kk in zip(keys, ks))
        if not need:
            continue
        try:
            ex = examples[idx]
            user, gold = build_user_and_gold(dataset_name, ex)
            if rec is None:
                r_o = extract_compact(tok, model, format_prompt(tok, user, think), cfg, stop_ids)
                pred_o = parse_pred(dataset_name, r_o["generated_text"])
                c_o = is_correct(dataset_name, pred_o, gold)
                fo, _ = earliest_correct_step(dataset_name, r_o["gen_ids"], tok, gold)
                rec = {"idx": idx, "dataset": dataset_name, "gold": str(gold),
                       "original": {"generated": r_o["generated_text"], "pred": str(pred_o), "correct": c_o,
                                    "hidden": r_o["hidden"], "metrics": r_o["metrics"],
                                    "gen_len": r_o["gen_len"], "first_correct_step": int(fo)},
                       "variants": {}}
                by_idx[idx] = rec
            o = rec["original"]
            c_o = o["correct"]

            for tf, vkey, k in zip(transforms, keys, ks):
                slot = rec["variants"].setdefault(vkey, {"transform": tf, "k": k, "samples": []})
                slot["k"] = max(slot.get("k", k), k)
                samples = slot["samples"]
                seen = {s.get("tf_question"): s for s in samples if s["applied"] and not s.get("duplicate")}
                for si in range(len(samples), k):
                    set_seed(variant_seed(idx, vkey, si))
                    user_tf, tmeta = build_transformed_user(dataset_name, ex, tf, llm_ctx,
                                                            idx=idx, sample=si, vkey=vkey, seed=seed)
                    qtf = tmeta.get("transformed_question", "")
                    base = {"sample": si, "guard": tmeta.get("guard"),
                            "variation": tmeta.get("variation"), "leak_flag": tmeta.get("leak_flag", False)}

                    if tmeta.get("num_changed", 0) == 0:
                        samples.append({**base, "applied": False, "duplicate": False,
                            "tf_question": qtf, "generated": o["generated"], "pred": o["pred"], "correct": c_o,
                            "flip_to_correct": False, "flip_to_incorrect": False,
                            "hidden": torch.empty(0), "metrics": torch.empty(0),
                            "gen_len": o["gen_len"], "first_correct_step": o["first_correct_step"]})
                        continue

                    if qtf in seen:
                        p = seen[qtf]
                        samples.append({**base, "applied": True, "duplicate": True, "tf_question": qtf,
                            "generated": p["generated"], "pred": p["pred"], "correct": p["correct"],
                            "flip_to_correct": p["flip_to_correct"], "flip_to_incorrect": p["flip_to_incorrect"],
                            "hidden": torch.empty(0), "metrics": torch.empty(0),
                            "gen_len": p["gen_len"], "first_correct_step": p["first_correct_step"]})
                        continue

                    r_t = extract_compact(tok, model, format_prompt(tok, user_tf, think), cfg, stop_ids)
                    pred_t = parse_pred(dataset_name, r_t["generated_text"])
                    c_t = is_correct(dataset_name, pred_t, gold)
                    ft, _ = earliest_correct_step(dataset_name, r_t["gen_ids"], tok, gold)
                    smp = {**base, "applied": True, "duplicate": False, "tf_question": qtf,
                           "generated": r_t["generated_text"], "pred": str(pred_t), "correct": c_t,
                           "flip_to_correct": (not c_o and c_t), "flip_to_incorrect": (c_o and not c_t),
                           "hidden": r_t["hidden"], "metrics": r_t["metrics"],
                           "gen_len": r_t["gen_len"], "first_correct_step": int(ft)}
                    samples.append(smp)
                    seen[qtf] = smp
            n_new += 1
            if n_new % cfg["save_every"] == 0:
                _flush()
            pbar.set_postfix(records=len(by_idx), new=n_new)
        except Exception as e:
            print(f"[skip idx {idx}] {type(e).__name__}: {e}")
        finally:
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    _flush()
    recs = list(by_idx.values())
    print(f"\nDone -> {drive_pt} | {len(recs)} examples")
    base = sum(r["original"]["correct"] for r in recs) / len(recs)
    print(f"  {'BASELINE (deterministic)':28s} acc {base:.2%}")
    for k in keys:
        rows = [(r["original"]["correct"], r["original"]["pred"], r["gold"],
                 [s for s in r["variants"][k]["samples"] if s["applied"]])
                for r in recs if k in r["variants"]]
        rows = [t for t in rows if t[3]]
        if not rows:
            print(f"  {k:28s} never applied")
            continue
        orig = sum(oc for oc, _, _, _ in rows) / len(rows)
        mean_k = sum(sum(s["correct"] for s in ss) / len(ss) for _, _, _, ss in rows) / len(rows)
        pass_k = sum(any(s["correct"] for s in ss) for _, _, _, ss in rows) / len(rows)
        vote_k = sum(_vote([s["pred"] for s in ss], op) == g for _, op, g, ss in rows) / len(rows)
        print(f"  {k:28s} n={len(rows):3d} | orig {orig:.2%} | mean@k {mean_k:.2%} ({mean_k-orig:+.2%})"
              f" | VOTE@k {vote_k:.2%} ({vote_k-orig:+.2%}) | pass@k {pass_k:.2%} ({pass_k-orig:+.2%})")
    del model, tok
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return recs, drive_pt

