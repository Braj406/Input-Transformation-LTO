"""Post-hoc analysis used on top of a trained TrajectoryTransformer + collected .pt records:
row/split construction, flip-count transitions, qualitative flip inspection, reward-gap
("confidence") diagnostics, and a feature-engineering + grouped-CV classifier sweep.

This module holds the analysis that notebook 2 (qwen3-1.7b run) added on top of the shared
pipeline/classifier/lto_algorithm modules; notebook 1 doesn't use it.
"""
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Diagnostic: how many usable (metrics-bearing) trajectories do we actually have?
# ---------------------------------------------------------------------------

def usable_row_report(recs, keys):
    def count_usable(vkey):
        total = usable = correct = 0
        for r in recs:
            v = r["variants"].get(vkey)
            if not v:
                continue
            for s in v["samples"]:
                total += 1
                if s["metrics"].numel() > 0:
                    usable += 1
                    correct += int(s["correct"])
        return total, usable, correct

    print(f"{'source':<20}{'total':>8}{'usable':>8}{'correct':>9}{'incorrect':>11}")
    grand_usable = grand_correct = 0

    orig_usable = sum(1 for r in recs if r["original"]["metrics"].numel() > 0)
    orig_correct = sum(int(r["original"]["correct"]) for r in recs if r["original"]["metrics"].numel() > 0)
    print(f"{'original':<20}{len(recs):>8}{orig_usable:>8}{orig_correct:>9}{orig_usable-orig_correct:>11}")
    grand_usable += orig_usable
    grand_correct += orig_correct

    for vkey in keys:
        total, usable, correct = count_usable(vkey)
        print(f"{vkey:<20}{total:>8}{usable:>8}{correct:>9}{usable-correct:>11}")
        grand_usable += usable
        grand_correct += correct

    print(f"\nGRAND TOTAL usable rows: {grand_usable}  (correct={grand_correct}, "
          f"incorrect={grand_usable-grand_correct}, correct rate={grand_correct/grand_usable:.1%})")
    print(f"Unique questions in this .pt: {len(recs)}")
    return grand_usable, grand_correct


# ---------------------------------------------------------------------------
# Row / split construction (shared prep for both the classifier and this analysis)
# ---------------------------------------------------------------------------

def collect_rows(recs, keys, include_original=True):
    rows = []
    if include_original:
        for r in recs:
            o = r["original"]
            if o["metrics"].numel() == 0:
                continue
            rows.append({"hidden": o["hidden"].float(), "metrics": o["metrics"].float(),
                         "label": float(o["correct"]), "source": "original", "idx": r["idx"]})
    for vkey in keys:
        for r in recs:
            v = r["variants"].get(vkey)
            if not v:
                continue
            for s in v["samples"]:
                if s["metrics"].numel() == 0:
                    continue
                rows.append({"hidden": s["hidden"].float(), "metrics": s["metrics"].float(),
                             "label": float(s["correct"]), "source": vkey, "idx": r["idx"]})
    return rows


def grouped_split(rows, seed=0, test_size=0.30, val_fraction_of_temp=0.5):
    """Split by QUESTION idx, not by row -- a question's original + all its transformed
    variants must land entirely in one split, or the model can leak per-question identity
    instead of learning a genuine correctness signal."""
    idxs = np.array([r["idx"] for r in rows])
    labels = np.array([r["label"] for r in rows])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_i, temp_i = next(gss1.split(rows, labels, groups=idxs))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_temp, random_state=seed)
    val_i_rel, test_i_rel = next(gss2.split(temp_i, labels[temp_i], groups=idxs[temp_i]))
    val_i, test_i = temp_i[val_i_rel], temp_i[test_i_rel]

    print(f"split (by question): train={len(train_i)} val={len(val_i)} test={len(test_i)}")
    print(f"correct-rate  train={labels[train_i].mean():.1%}  val={labels[val_i].mean():.1%}  "
          f"test={labels[test_i].mean():.1%}")
    return idxs, labels, train_i, val_i, test_i


def standardize_metrics(rows, train_i):
    """Standardize metrics using TRAIN stats only, per (layer, metric)."""
    metrics_stack = torch.stack([r["metrics"] for r in rows])          # [N, L, 4]
    m_mean = metrics_stack[train_i].mean(dim=0, keepdim=True)
    m_std = metrics_stack[train_i].std(dim=0, keepdim=True).clamp_min(1e-6)
    return (metrics_stack - m_mean) / m_std


def maybe_pca_hidden(rows, train_i, use_pca=False, pca_dim=24, seed=0):
    """Optionally compress hidden states before feeding the transformer, to avoid a huge
    learned raw_hidden_dim->d_model projection overfitting on a small dataset. PCA is fit
    on TRAIN rows only."""
    hidden_stack = torch.stack([r["hidden"] for r in rows])            # [N, L, RAW_HIDDEN_DIM]
    if not use_pca:
        return hidden_stack, None
    raw_dim = hidden_stack.shape[-1]
    L = hidden_stack.shape[1]
    train_hidden_flat = hidden_stack[train_i].reshape(-1, raw_dim).numpy()
    pca = PCA(n_components=pca_dim, random_state=seed).fit(train_hidden_flat)
    n_all = hidden_stack.shape[0]
    hidden_flat = hidden_stack.reshape(-1, raw_dim).numpy()
    hidden_stack = torch.tensor(pca.transform(hidden_flat).reshape(n_all, L, pca_dim), dtype=torch.float32)
    print(f"PCA: {raw_dim} -> {pca_dim} dims  (explained variance: {pca.explained_variance_ratio_.sum():.1%})")
    return hidden_stack, pca


# ---------------------------------------------------------------------------
# Flip-count transitions: original (untransformed) answer vs. classifier-selected candidate
# ---------------------------------------------------------------------------

def compute_transitions(selected_correct_by_idx, orig_correct_by_idx):
    o2c = c2o = o2o = c2c = 0
    for idx, sel_c in selected_correct_by_idx.items():
        oc = orig_correct_by_idx[idx]
        if not oc and sel_c:
            o2c += 1
        elif oc and not sel_c:
            c2o += 1
        elif oc and sel_c:
            c2c += 1
        else:
            o2o += 1
    return dict(incorrect_to_correct=o2c, correct_to_incorrect=c2o,
                stayed_correct=c2c, stayed_incorrect=o2o)


@torch.no_grad()
def greedy_correct_by_task(model, hidden_stack, metrics_stack, label_stack, tasks_grouped, device):
    model.eval()
    out = {}
    for q_idx, row_positions in tasks_grouped.items():
        h = hidden_stack[row_positions].to(device)
        m_ = metrics_stack[row_positions].to(device)
        probs = torch.sigmoid(model(h, m_)).cpu().numpy()
        greedy_local_idx = int(np.argmax(probs))
        out[q_idx] = bool(label_stack[row_positions[greedy_local_idx]].item())
    return out


def mcnemar_on_transitions(trans):
    b, c = trans["incorrect_to_correct"], trans["correct_to_incorrect"]
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n > 0 else float("nan")
    return b, c, n, p


def lto_transitions_across_seeds(model, hidden_stack, metrics_stack, label_stack, tasks_grouped,
                                 orig_correct_by_idx, device, beta=0.05, n_seeds=10):
    from .lto_algorithm import conduct_rejection_sampling
    model.eval()
    runs = []
    with torch.no_grad():
        for seed in range(n_seeds):
            np.random.seed(seed)
            selected = {}
            for q_idx, row_positions in tasks_grouped.items():
                N = len(row_positions)
                h = hidden_stack[row_positions].to(device)
                m_ = metrics_stack[row_positions].to(device)
                probs = torch.sigmoid(model(h, m_)).cpu().numpy()
                li = conduct_rejection_sampling(list(range(N)), probs.tolist(), 1, beta=beta)[0]
                selected[q_idx] = bool(label_stack[row_positions[li]].item())
            runs.append(compute_transitions(selected, orig_correct_by_idx))

    def mean_std(key):
        vals = np.array([t[key] for t in runs])
        return vals.mean(), vals.std()

    print(f"\nLTO vs ORIGINAL (mean +/- std over {n_seeds} seeds):")
    for key, label in [("incorrect_to_correct", "incorrect -> correct"),
                        ("correct_to_incorrect", "correct -> incorrect"),
                        ("stayed_correct", "stayed correct"),
                        ("stayed_incorrect", "stayed incorrect")]:
        m, s = mean_std(key)
        print(f"  {label:<22}: {m:.1f} +/- {s:.1f}")
    net = np.array([t["incorrect_to_correct"] - t["correct_to_incorrect"] for t in runs])
    print(f"  net change            : {net.mean():+.1f} +/- {net.std():.1f} tasks")
    return runs


# ---------------------------------------------------------------------------
# Qualitative inspection: what did Greedy actually pick, and does it look wrong?
# ---------------------------------------------------------------------------

def build_candidates(q_idx, recs_by_idx, examples_ds, classifier_keys):
    """Same candidate set/order as collect_rows -- original first, then each
    classifier_keys in order -- but keeps the actual text/pred instead of stripping it."""
    q_idx = int(q_idx)
    r = recs_by_idx[q_idx]
    cands = []
    o = r["original"]
    if o["metrics"].numel() > 0:
        cands.append(dict(source="original", text=examples_ds[q_idx]["question"],
                          pred=o["pred"], correct=bool(o["correct"]),
                          hidden=o["hidden"], metrics=o["metrics"]))
    for vkey in classifier_keys:
        v = r["variants"].get(vkey)
        if not v:
            continue
        for s in v["samples"]:
            if s["metrics"].numel() == 0:
                continue
            cands.append(dict(source=vkey, text=s.get("tf_question"),
                              pred=s["pred"], correct=bool(s["correct"]),
                              hidden=s["hidden"], metrics=s["metrics"]))
    return r, cands


def inspect_task(q_idx, model, recs_by_idx, examples_ds, classifier_keys, device):
    r, cands = build_candidates(q_idx, recs_by_idx, examples_ds, classifier_keys)
    h = torch.stack([c["hidden"] for c in cands]).to(device)
    m_ = torch.stack([c["metrics"] for c in cands]).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(h, m_)).cpu().numpy()
    greedy_i = int(np.argmax(probs))

    print(f"\n{'='*100}\n[idx {int(q_idx)}]  gold={r['gold']}")
    print(f"ORIGINAL: {examples_ds[int(q_idx)]['question']}  "
          f"(pred={r['original']['pred']}, correct={r['original']['correct']})")
    print(f"{'source':<20}{'reward':>8}{'pred':>6}{'correct':>9}   text")
    for c, p in zip(cands, probs):
        tag = "  <== GREEDY PICK" if c is cands[greedy_i] else ""
        txt = (c["text"][:70] + "...") if c["text"] and len(c["text"]) > 70 else c["text"]
        print(f"{c['source']:<20}{p:>8.3f}{str(c['pred']):>6}{str(c['correct']):>9}   {txt}{tag}")


# ---------------------------------------------------------------------------
# Reward compression: does the model actually discriminate between candidates?
# ---------------------------------------------------------------------------

@torch.no_grad()
def reward_gap_stats(model, hidden_stack, metrics_stack, tasks_grouped, device):
    gaps, all_probs = [], []
    model.eval()
    for q_idx, row_positions in tasks_grouped.items():
        h = hidden_stack[row_positions].to(device)
        m_ = metrics_stack[row_positions].to(device)
        probs = torch.sigmoid(model(h, m_)).cpu().numpy()
        all_probs.extend(probs.tolist())
        if len(probs) > 1:
            sorted_p = np.sort(probs)[::-1]
            gaps.append(sorted_p[0] - sorted_p[1])

    all_probs, gaps = np.array(all_probs), np.array(gaps)
    print(f"reward range across ALL {len(all_probs)} test candidates: "
          f"min={all_probs.min():.3f}  max={all_probs.max():.3f}  mean={all_probs.mean():.3f}  "
          f"std={all_probs.std():.3f}")
    print(f"gap between top-pick and runner-up per task: "
          f"mean={gaps.mean():.4f}  median={np.median(gaps):.4f}  max={gaps.max():.4f}")
    print(f"tasks decided by a gap < 0.01 (essentially a coin flip): "
          f"{(gaps < 0.01).sum()}/{len(gaps)} ({(gaps < 0.01).mean():.1%})")
    return all_probs, gaps


@torch.no_grad()
def confidence_bucket_stats(model, hidden_stack, metrics_stack, label_stack, tasks_grouped,
                            orig_correct_by_idx, device, gap_threshold=0.01):
    """Does the accuracy lift concentrate in high-confidence tasks vs coin-flip tasks?"""
    model.eval()
    task_gap, task_greedy_correct = {}, {}
    for q_idx, row_positions in tasks_grouped.items():
        h = hidden_stack[row_positions].to(device)
        m_ = metrics_stack[row_positions].to(device)
        probs = torch.sigmoid(model(h, m_)).cpu().numpy()
        sorted_p = np.sort(probs)[::-1]
        task_gap[q_idx] = sorted_p[0] - sorted_p[1] if len(probs) > 1 else float("inf")
        task_greedy_correct[q_idx] = bool(label_stack[row_positions[int(np.argmax(probs))]].item())

    high_conf = [idx for idx in tasks_grouped if task_gap[idx] >= gap_threshold]
    coin_flip = [idx for idx in tasks_grouped if task_gap[idx] < gap_threshold]

    def bucket_stats(bucket):
        n = len(bucket)
        orig_acc = np.mean([orig_correct_by_idx[i] for i in bucket])
        greedy_acc_b = np.mean([task_greedy_correct[i] for i in bucket])
        o2c = sum(1 for i in bucket if not orig_correct_by_idx[i] and task_greedy_correct[i])
        c2o = sum(1 for i in bucket if orig_correct_by_idx[i] and not task_greedy_correct[i])
        return dict(n=n, orig_acc=orig_acc, greedy_acc=greedy_acc_b, o2c=o2c, c2o=c2o)

    results = {}
    for name, bucket in [("high_confidence", high_conf), ("coin_flip", coin_flip)]:
        s = bucket_stats(bucket)
        results[name] = s
        label = f"HIGH-CONFIDENCE (gap>={gap_threshold})" if name == "high_confidence" \
            else f"COIN-FLIP (gap<{gap_threshold})"
        print(f"{label}: n={s['n']:2d}  orig_acc={s['orig_acc']:.1%}  greedy_acc={s['greedy_acc']:.1%}  "
              f"net_flips: +{s['o2c']} -{s['c2o']} = {s['o2c']-s['c2o']:+d}")
    return results


# ---------------------------------------------------------------------------
# Feature engineering + honest grouped-CV classifier sweep (logistic / RF / hist-GB
# over per-layer metrics, deltas from the original, gen-length, and source one-hot).
# Test questions are held out BEFORE any sweeping; CV runs only on dev questions.
# ---------------------------------------------------------------------------

def build_feature_table(recs, keys):
    orig_metrics = {r["idx"]: r["original"]["metrics"].float()
                    for r in recs if r["original"]["metrics"].numel() > 0}
    orig_genlen = {r["idx"]: r["original"]["gen_len"] for r in recs}
    sources = ["original"] + list(keys)

    raw = []
    for r in recs:
        o = r["original"]
        if o["metrics"].numel() > 0:
            raw.append(dict(idx=r["idx"], source="original", metrics=o["metrics"].float(),
                            gen_len=o["gen_len"], label=float(o["correct"])))
        for vkey in keys:
            v = r["variants"].get(vkey)
            if not v:
                continue
            for s in v["samples"]:
                if s["metrics"].numel() == 0:
                    continue
                raw.append(dict(idx=r["idx"], source=vkey, metrics=s["metrics"].float(),
                                gen_len=s["gen_len"], label=float(s["correct"])))

    F, y, g, src = [], [], [], []
    for row in raw:
        m = row["metrics"].numpy().ravel()                                             # per-layer metrics
        om = orig_metrics.get(row["idx"])
        d = (row["metrics"] - om).numpy().ravel() if om is not None else np.zeros_like(m)  # delta from orig
        gl = np.log1p(row["gen_len"])
        gld = gl - np.log1p(orig_genlen.get(row["idx"], row["gen_len"]))
        oh = [1.0 if row["source"] == s else 0.0 for s in sources]
        F.append(np.concatenate([m, d, [gl, gld], oh]))
        y.append(row["label"])
        g.append(row["idx"])
        src.append(row["source"])

    n_m = len(raw[0]["metrics"].numpy().ravel())
    slices = {"metrics": slice(0, n_m), "delta": slice(n_m, 2 * n_m),
              "genlen": slice(2 * n_m, 2 * n_m + 2), "source": slice(2 * n_m + 2, None)}
    return np.array(F, np.float32), np.array(y), np.array(g), np.array(src), slices


def _subset(X, sl_list):
    return np.concatenate([X[:, s] for s in sl_list], axis=1)


def cv_eval(make_clf, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs = [], []
    for tr, te in gkf.split(X, y, groups):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        p_tr, p_te = clf.predict_proba(X[tr])[:, 1], clf.predict_proba(X[te])[:, 1]
        cand = np.unique(np.round(p_tr, 3))                       # threshold tuned on TRAIN fold only
        thr = max(cand, key=lambda t: accuracy_score(y[tr], p_tr >= t))
        aucs.append(roc_auc_score(y[te], p_te))
        accs.append(accuracy_score(y[te], p_te >= thr))
    return np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs)


def default_classifier_grid(seed=0):
    grid = []
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        for cw in [None, "balanced"]:
            grid.append((f"LR C={C} cw={cw}",
                        lambda C=C, cw=cw: make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=C, class_weight=cw))))
    for md in [3, 4, 6, None]:
        for msl in [1, 5, 15]:
            grid.append((f"RF depth={md} leaf={msl}",
                        lambda md=md, msl=msl: RandomForestClassifier(n_estimators=400, max_depth=md,
                            min_samples_leaf=msl, class_weight="balanced", random_state=seed, n_jobs=-1)))
    for md in [2, 3, None]:
        for lr_ in [0.03, 0.1]:
            grid.append((f"HGB depth={md} lr={lr_}",
                        lambda md=md, lr_=lr_: HistGradientBoostingClassifier(max_depth=md,
                            learning_rate=lr_, max_iter=300, l2_regularization=1.0, random_state=seed)))
    return grid


def run_feature_sweep(recs, keys, seed=0, test_size=0.20, grid=None):
    X_all, y_all, groups_all, src_all, sl = build_feature_table(recs, keys)
    print(f"feature table: {X_all.shape}  |  positives {y_all.mean():.1%}  |  "
          f"questions {len(set(groups_all))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_i, test_i_final = next(gss.split(X_all, y_all, groups=groups_all))
    print(f"dev rows={len(dev_i)} ({len(set(groups_all[dev_i]))} questions) | "
          f"test rows={len(test_i_final)} ({len(set(groups_all[test_i_final]))} questions)")

    feature_sets = {
        "metrics only":       [sl["metrics"]],
        "metrics+delta":      [sl["metrics"], sl["delta"]],
        "metrics+delta+len":  [sl["metrics"], sl["delta"], sl["genlen"]],
        "all (incl. source)": [sl["metrics"], sl["delta"], sl["genlen"], sl["source"]],
    }
    grid = grid or default_classifier_grid(seed)

    Xd, yd, gd = X_all[dev_i], y_all[dev_i], groups_all[dev_i]
    base = max(yd.mean(), 1 - yd.mean())
    print(f"\nmajority baseline (dev): {base:.1%}\n")

    results = []
    for fs_name, sl_list in feature_sets.items():
        Xf = _subset(Xd, sl_list)
        print(f"--- {fs_name}  ({Xf.shape[1]} features) ---")
        best = None
        for name, mk in grid:
            auc_m, auc_s, acc_m, acc_s = cv_eval(mk, Xf, yd, gd)
            results.append((fs_name, name, auc_m, auc_s, acc_m, acc_s))
            if best is None or auc_m > best[2]:
                best = (fs_name, name, auc_m, auc_s, acc_m, acc_s)
        print(f"  BEST: {best[1]:<26} AUC {best[2]:.3f}+/-{best[3]:.3f}  acc {best[4]:.1%}+/-{best[5]:.1%}\n")

    print("=" * 78)
    print("TOP 10 CONFIGS OVERALL (5-fold grouped CV on dev)")
    print("=" * 78)
    for fs, name, am, asd, cm, csd in sorted(results, key=lambda r: -r[2])[:10]:
        print(f"{fs:<22}{name:<26} AUC {am:.3f}+/-{asd:.3f}  acc {cm:.1%}+/-{csd:.1%}")

    return X_all, y_all, groups_all, src_all, sl, dev_i, test_i_final, results
