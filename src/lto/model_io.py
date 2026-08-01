"""Tokenizer/model loading, chat-template formatting, and stop-token resolution."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ATTN, DEVICE, DTYPE, LOAD_IN_4BIT, MODELS


def format_prompt(tokenizer, user_content, enable_thinking=None):
    if tokenizer.chat_template:
        msgs = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}))
        except TypeError:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_content + "\nAnswer:"


def stop_token_ids(tok, model):
    stop = set()
    if tok.eos_token_id is not None:
        stop.add(tok.eos_token_id)
    ge = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(ge, int):
        stop.add(ge)
    elif isinstance(ge, (list, tuple)):
        stop.update(ge)
    return stop


def load_model(model_key):
    spec = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(spec["id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = dict(torch_dtype=DTYPE, low_cpu_mem_usage=True, attn_implementation=ATTN)
    if LOAD_IN_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(spec["id"], device_map={"": 0}, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec["id"], **kw).to(DEVICE)
    model.eval()
    return tok, model, spec


@torch.inference_mode()
def sanity_check(tok, model, spec):
    L, H = model.config.num_hidden_layers, model.config.hidden_size
    if spec.get("layers") is not None and L != spec["layers"]:
        print(f"[sanity] NOTE: config reports {L} layers (registry said {spec['layers']}); using {L}.")
    enc = tok("Sanity check.", return_tensors="pt").to(model.device)
    o = model(**enc, output_hidden_states=True)
    assert len(o.hidden_states) == L + 1
    assert tok.eos_token_id is not None
    print(f"[sanity] {spec['id']} layers={L} hidden={H} hidden_states={L+1} eos={tok.eos_token_id} OK")
    return L, H
