"""Global run configuration: seeding, device/dtype selection, and the model registry."""
import random

import numpy as np
import torch

SEED = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
ATTN = "sdpa"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
LOAD_IN_4BIT = False

MODELS = {
    "qwen3-0.6b":   {"id": "Qwen/Qwen3-0.6B",                     "short": "qwen3_0p6b",  "layers": 28},
    "qwen3-1.7b":   {"id": "Qwen/Qwen3-1.7B",                     "short": "qwen3_1p7b",  "layers": 28},
    "qwen3-4b":     {"id": "Qwen/Qwen3-4B",                       "short": "qwen3",       "layers": 36},
    "qwen2.5-0.5b": {"id": "Qwen/Qwen2.5-0.5B-Instruct",          "short": "qwen25_0p5b", "layers": 24},
    "qwen2.5-1.5b": {"id": "Qwen/Qwen2.5-1.5B-Instruct",          "short": "qwen25_1p5b", "layers": 28},
    "smollm2-1.7b": {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "short": "smollm2_1p7b", "layers": 24},
    "llama3.2-1b":  {"id": "meta-llama/Llama-3.2-1B-Instruct",    "short": "llama32_1b",  "layers": 16},
    "llama3.2-3b":  {"id": "meta-llama/Llama-3.2-3B-Instruct",    "short": "llama32_3b",  "layers": 28},
    "llama3-8b":    {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "short": "llama3",      "layers": 32},
}

COMPACT_CFG = {
    "max_new_tokens":        256,
    "max_metric_tokens":     500,
    "hidden_pooling":        "mean",
    "enable_thinking":       False,
    "save_every":            5,
    # Answer inference is deterministic: temperature 0 == pure argmax (see extraction.extract_compact).
    "inference_temperature": 0.0,
    "inference_greedy":      True,
}
assert COMPACT_CFG["inference_temperature"] == 0.0 and COMPACT_CFG["inference_greedy"], \
    "Answer inference must stay deterministic (temperature 0 / greedy)."
