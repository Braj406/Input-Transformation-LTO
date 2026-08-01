"""Representation-geometry metrics computed over a trajectory's per-layer hidden states."""
import math

import torch

METRIC_NAMES = ["entropy", "effective_rank", "anisotropy", "intrinsic_dimension"]


def compute_matrix_entropy(layer_representations):
    with torch.no_grad():
        X = layer_representations.to(torch.float32)
        G = torch.matmul(X, X.T)
        tr = torch.trace(G)
        if tr <= 1e-9:
            return 0.0
        G = G / tr
        ev = torch.linalg.eigvalsh(G)
        ev = ev[ev > 1e-9]
        return (-torch.sum(ev * torch.log(ev))).item()


def calculate_effective_rank(layer_representations):
    with torch.no_grad():
        X = layer_representations.to(torch.float32)
        try:
            _, s, _ = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            return 1.0
        s = s[s > 1e-9]
        tot = torch.sum(s)
        if tot <= 1e-9:
            return 1.0
        p = s / tot
        return math.exp((-torch.sum(p * torch.log(p))).item())


def calculate_anisotropy(layer_representations):
    with torch.no_grad():
        X = layer_representations.to(torch.float32)
        Xc = X - X.mean(0, keepdim=True)
        C = torch.matmul(Xc, Xc.T)
        ev = torch.linalg.svdvals(C)
        tot = torch.sum(ev)
        if tot <= 1e-9:
            return 1.0
        return (ev[0] / tot).item()


def calculate_intrinsic_dimension(layer_representations, trimming_factor=0.1):
    with torch.no_grad():
        X = layer_representations.to(torch.float32)
        n = X.shape[0]
        if n < 3:
            return 0.0
        D = torch.cdist(X, X, p=2)
        r1, r2 = [], []
        for i in range(n):
            sd, _ = torch.sort(D[i])
            r1.append(sd[1])
            r2.append(sd[2])
        r1 = torch.clamp(torch.stack(r1), min=1e-7)
        r2 = torch.stack(r2)
        mu, _ = torch.sort(r2 / r1)
        xs, ys = [], []
        for rank in range(1, n + 1):
            F = rank / n
            xs.append(math.log(max(mu[rank - 1].item(), 1e-9)))
            ys.append(-math.log(1 - F + 1e-9))
        if trimming_factor > 0:
            k = int((1 - trimming_factor) * n)
            xs, ys = xs[:k], ys[:k]
        xt, yt = torch.tensor(xs), torch.tensor(ys)
        den = torch.sum(xt ** 2)
        if den == 0:
            return 0.0
        return (torch.sum(xt * yt) / den).item()
