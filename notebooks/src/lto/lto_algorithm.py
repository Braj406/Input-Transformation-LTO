"""Latent Thinking Optimization (LTO) Algorithm 1: reward-guided rejection sampling over
candidate trajectories, using a trained TrajectoryTransformer as the reward model."""
from collections import defaultdict

import numpy as np
import torch


def conduct_rejection_sampling(response_candidates, response_rewards, num_samples, beta=0.05):
    """Acceptance-Rejection sampler (LTO Algorithm 1)."""
    candidates = {c: r for c, r in zip(range(len(response_candidates)), response_rewards)}
    accepted = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            to_remove.append(c)
            if len(accepted) == num_samples:
                break
        for c in to_remove:
            candidates.pop(c)
    return [response_candidates[idx] for idx in accepted]


def group_tasks_by_question(idxs, split_row_positions):
    """Group row positions (e.g. a test split's row indices) by question idx -- each
    question's original + all its transformed variants form one "task" (candidate set)."""
    grouped = defaultdict(list)
    for row_pos in split_row_positions:
        grouped[idxs[row_pos]].append(row_pos)
    return grouped


@torch.no_grad()
def run_lto_algorithm1(tasks_grouped, hidden_stack, metrics_stack, label_stack, model, device,
                        beta=0.05, seed=0, verbose=True):
    """Runs Algorithm 1 once (one rejection-sampling draw per task) and reports
    Base/LTO/Greedy/Oracle Pass@1, plus the selection confusion matrix."""
    np.random.seed(seed)
    model.eval()
    base_correct = lto_correct = greedy_correct = oracle_correct = 0.0
    total_tasks = 0
    lto_actuals, lto_decisions = [], []

    for q_idx, row_positions in tasks_grouped.items():
        N = len(row_positions)
        if N == 0:
            continue
        task_labels = [float(label_stack[p].item()) for p in row_positions]

        base_correct += sum(task_labels) / N
        oracle_correct += 1 if max(task_labels) == 1.0 else 0
        total_tasks += 1
        lto_actuals.extend(task_labels)

        h = hidden_stack[row_positions].to(device)
        m_ = metrics_stack[row_positions].to(device)
        probs = torch.sigmoid(model(h, m_)).cpu().numpy()

        lto_local_idx = conduct_rejection_sampling(list(range(N)), probs.tolist(), 1, beta=beta)[0]
        greedy_local_idx = int(np.argmax(probs))

        if task_labels[lto_local_idx] == 1.0:
            lto_correct += 1
        if task_labels[greedy_local_idx] == 1.0:
            greedy_correct += 1

        dec = [0] * N
        dec[lto_local_idx] = 1
        lto_decisions.extend(dec)

    base_acc = 100 * base_correct / total_tasks
    lto_acc = 100 * lto_correct / total_tasks
    greedy_acc = 100 * greedy_correct / total_tasks
    oracle_acc = 100 * oracle_correct / total_tasks

    lto_a, lto_d = np.array(lto_actuals), np.array(lto_decisions)
    matrix = {
        "true_positive": int(np.sum((lto_a == 1) & (lto_d == 1))),
        "false_negative": int(np.sum((lto_a == 1) & (lto_d == 0))),
        "false_positive": int(np.sum((lto_a == 0) & (lto_d == 1))),
        "true_negative": int(np.sum((lto_a == 0) & (lto_d == 0))),
    }
    precision = matrix["true_positive"] / (matrix["true_positive"] + matrix["false_positive"] + 1e-9)

    if verbose:
        print("═" * 60)
        print(f"Total Test Tasks Evaluated: {total_tasks}")
        print(f"Base Model Pass@1:          {base_acc:.2f}%  (random pick, ~= mean@k)")
        print(f"LTO Optimized Pass@1:       {lto_acc:.2f}%  (Algorithm 1)")
        print(f"Greedy Pass@1:              {greedy_acc:.2f}%  (argmax reward)")
        print(f"Oracle Pass@1:              {oracle_acc:.2f}%  (upper bound, ~= pass@k)")
        print(f"Absolute Improvement:       {lto_acc - base_acc:+.2f}%  "
              f"(headroom to oracle: {oracle_acc - base_acc:+.2f}%)")
        print("═" * 60)
        print(f"Correct Trajectory (Actual 1)   -> LTO Accepted (1): {matrix['true_positive']}")
        print(f"Correct Trajectory (Actual 1)   -> LTO Rejected (0): {matrix['false_negative']}")
        print(f"Incorrect Trajectory (Actual 0) -> LTO Accepted (1): {matrix['false_positive']}")
        print(f"Incorrect Trajectory (Actual 0) -> LTO Rejected (0): {matrix['true_negative']}")
        print(f"LTO Selection Precision:      {precision:.2%} (the final Pass@1)")
        print("═" * 60)

    return {
        "total_tasks": total_tasks, "base_acc": base_acc, "lto_acc": lto_acc,
        "greedy_acc": greedy_acc, "oracle_acc": oracle_acc,
        "selection_matrix": matrix, "selection_precision": precision,
    }


def lto_stability_across_seeds(tasks_grouped, hidden_stack, metrics_stack, label_stack, model, device,
                                beta=0.05, n_seeds=10):
    """LTO's rejection sampling is stochastic; Base/Greedy/Oracle are deterministic and
    don't need re-running. Reports mean/std Pass@1 across seeds."""
    model.eval()
    total_tasks = len(tasks_grouped)
    accs = []
    with torch.no_grad():
        for seed in range(n_seeds):
            np.random.seed(seed)
            lc = 0
            for q_idx, row_positions in tasks_grouped.items():
                N = len(row_positions)
                task_labels = [float(label_stack[p].item()) for p in row_positions]
                h = hidden_stack[row_positions].to(device)
                m_ = metrics_stack[row_positions].to(device)
                probs = torch.sigmoid(model(h, m_)).cpu().numpy()
                li = conduct_rejection_sampling(list(range(N)), probs.tolist(), 1, beta=beta)[0]
                if task_labels[li] == 1.0:
                    lc += 1
            accs.append(100 * lc / total_tasks)
    accs = np.array(accs)
    print(f"LTO Pass@1 across {n_seeds} seeds: mean={accs.mean():.2f}%  std={accs.std():.2f}%")
    return accs
