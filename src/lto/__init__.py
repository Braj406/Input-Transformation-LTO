"""LTO (Latent Thinking Optimization): generate reasoning trajectories under question
transforms, extract per-layer representation-geometry metrics, train a reward classifier
over trajectories, and select among candidates via rejection sampling.

Everything here is shared between the two experiment notebooks (notebooks/); each notebook
only sets its own experiment config (model, dataset, whether the classifier uses raw
hidden states) and calls into this package.
"""
from .config import COMPACT_CFG, DEVICE, DTYPE, GPU_NAME, MODELS, SEED, set_seed
from .nltk_setup import ensure_nltk_data
from .pipeline import run_multi

__all__ = [
    "COMPACT_CFG", "DEVICE", "DTYPE", "GPU_NAME", "MODELS", "SEED", "set_seed",
    "ensure_nltk_data", "run_multi",
]

__version__ = "0.1.0"
