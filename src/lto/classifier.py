"""Two-layer Transformer encoder over a trajectory's per-layer (hidden, metrics) tokens,
pooled via a CLS token, trained to predict correct vs. incorrect."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import DEVICE


class LayerTokenEncoder(nn.Module):
    def __init__(self, hidden_dim, n_metrics, d_model, use_hidden=True):
        super().__init__()
        self.use_hidden = use_hidden
        self.hidden_norm = nn.LayerNorm(hidden_dim) if use_hidden else None
        in_dim = (hidden_dim if use_hidden else 0) + n_metrics
        self.input_proj = nn.Linear(in_dim, d_model)

    def forward(self, hidden, metrics):
        x = torch.cat([self.hidden_norm(hidden), metrics], dim=-1) if self.use_hidden else metrics
        return self.input_proj(x)


class TrajectoryTransformer(nn.Module):
    def __init__(self, n_layers_seq, hidden_dim, n_metrics, d_model=32, nhead=2,
                 dim_feedforward=64, dropout=0.5, use_hidden=True):
        super().__init__()
        self.token_encoder = LayerTokenEncoder(hidden_dim, n_metrics, d_model, use_hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_layers_seq + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                        dim_feedforward=dim_feedforward, dropout=dropout,
                        batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, hidden, metrics):
        tok = self.token_encoder(hidden, metrics)               # [B, L, d_model]
        cls = self.cls.expand(tok.shape[0], -1, -1)
        x = torch.cat([cls, tok], dim=1) + self.pos_emb          # [B, L+1, d_model]
        x = self.encoder(x)
        return self.head(x[:, 0, :]).squeeze(-1)                 # logit from CLS token


def make_model(n_layers_seq, hidden_dim, n_metrics, d_model=32, nhead=2, dim_feedforward=64,
               dropout=0.5, use_hidden=True, device=None):
    device = device or DEVICE
    return TrajectoryTransformer(n_layers_seq=n_layers_seq, hidden_dim=hidden_dim, n_metrics=n_metrics,
                                 d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                 dropout=dropout, use_hidden=use_hidden).to(device)


def make_loader(hidden_stack, metrics_stack, label_stack, indices, batch_size, shuffle):
    ds = TensorDataset(hidden_stack[indices], metrics_stack[indices], label_stack[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(m, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    m.train() if train else m.eval()
    tot_loss, all_logits, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for h, mt, y in loader:
            h, mt, y = h.to(device), mt.to(device), y.to(device)
            logits = m(h, mt)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item() * len(y)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.cpu())
    logits, labs = torch.cat(all_logits), torch.cat(all_labels)
    probs = torch.sigmoid(logits)
    auc = roc_auc_score(labs, probs) if labs.unique().numel() > 1 else float("nan")
    acc = accuracy_score(labs, (probs > 0.5).float())
    return tot_loss / len(loader.dataset), auc, acc


def train_once(model_kwargs, train_loader, val_loader, label_stack, train_i, seed, lr,
               device=None, epochs=200, patience=25, verbose=False):
    device = device or DEVICE
    torch.manual_seed(seed)
    m = make_model(device=device, **model_kwargs)
    n_pos = label_stack[train_i].sum().item()
    n_neg = len(train_i) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_val_auc, best_state, bad_epochs, epoch = -1, None, 0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_auc, tr_acc = run_epoch(m, train_loader, criterion, device, optimizer)
        val_loss, val_auc, val_acc = run_epoch(m, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_auc)

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc, bad_epochs = val_auc, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch % 10 == 0 or improved):
            print(f"    epoch {epoch:3d} | val loss {val_loss:.3f} acc {val_acc:.1%} auc {val_auc:.3f}"
                  f"{'  *' if improved else ''}")

        if bad_epochs >= patience:
            break

    m.load_state_dict(best_state)
    return best_state, best_val_auc, epoch


def sweep_learning_rates(model_kwargs, train_loader, val_loader, label_stack, train_i, lr_grid,
                          n_runs_per_lr=10, base_seed=0, device=None, epochs=200, patience=25):
    """Train N independently-seeded runs per LR; report mean/std/min/max per LR, and
    return the best single run at the LR with the best MEAN val AUC (avoids cherry-picking
    a lucky run)."""
    device = device or DEVICE
    all_results = {}
    for lr in lr_grid:
        print(f"\n--- lr={lr} ---")
        runs = []
        for run_i in range(n_runs_per_lr):
            seed = base_seed + run_i
            state, auc, stopped_epoch = train_once(model_kwargs, train_loader, val_loader, label_stack,
                                                    train_i, seed, lr, device, epochs, patience, verbose=False)
            runs.append({"seed": seed, "auc": auc, "state": state, "epoch": stopped_epoch})
            print(f"  run {run_i+1:2d}/{n_runs_per_lr}  seed={seed}  stopped@epoch={stopped_epoch:3d}"
                  f"  best val AUC={auc:.3f}")
        aucs = np.array([r["auc"] for r in runs])
        print(f"  lr={lr}  mean={aucs.mean():.3f}  std={aucs.std():.3f}  min={aucs.min():.3f}  max={aucs.max():.3f}")
        all_results[lr] = runs

    print(f"\n{'='*60}\nSUMMARY across {len(lr_grid)} learning rates x {n_runs_per_lr} runs each")
    print(f"{'='*60}")
    print(f"{'lr':>10}{'mean AUC':>12}{'std':>8}{'min':>8}{'max':>8}")
    for lr, runs in all_results.items():
        aucs = np.array([r["auc"] for r in runs])
        print(f"{lr:>10}{aucs.mean():>12.3f}{aucs.std():>8.3f}{aucs.min():>8.3f}{aucs.max():>8.3f}")

    best_lr = max(all_results, key=lambda lr: np.mean([r["auc"] for r in all_results[lr]]))
    best_run_at_best_lr = max(all_results[best_lr], key=lambda r: r["auc"])
    model = make_model(device=device, **model_kwargs)
    model.load_state_dict(best_run_at_best_lr["state"])
    mean_at_best_lr = np.mean([r["auc"] for r in all_results[best_lr]])
    std_at_best_lr = np.std([r["auc"] for r in all_results[best_lr]])

    print(f"\nBest LR by mean across runs: {best_lr}  (mean={mean_at_best_lr:.3f}±{std_at_best_lr:.3f})")
    print(f"Loaded best single run at that LR into `model` (val AUC={best_run_at_best_lr['auc']:.3f}).")
    print(f"For reporting: use lr={best_lr}'s mean±std, not the single best run.")
    return model, all_results, best_lr
