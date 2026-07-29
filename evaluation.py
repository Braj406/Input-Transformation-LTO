import re
import torch
from metrics import (
    compute_matrix_entropy, 
    calculate_effective_rank, 
    calculate_anisotropy, 
    calculate_intrinsic_dimension
)

def parse_cqa(text):
    m = list(re.finditer(r"\b([A-E])\b", text))
    return m[-1].group(1).upper() if m else None

def parse_gsm(text):
    m = list(re.finditer(r"####\s*(-?[\d,]+(?:\.\d+)?)", text))
    if m: return float(m[-1].group(1).replace(",", ""))
    return None

def is_correct(name, pred, gold):
    if pred is None or gold is None: return False
    return pred == gold if name == "commonsense_qa" else abs(pred - gold) < 1e-4

@torch.inference_mode()
def extract_compact(tok, model, prompt_text, cfg, stop_ids):
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
        tid = int(nid)
        if tid in stop_ids: break
        gen_ids.append(tid)
        
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], 1)
        out = model(
            input_ids=nid.view(1, 1), attention_mask=attn, past_key_values=past,
            use_cache=True, output_hidden_states=True
        )
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
            rows.append([
                compute_matrix_entropy(M), 
                calculate_effective_rank(M),
                calculate_anisotropy(M), 
                calculate_intrinsic_dimension(M)
            ])
        metrics = torch.tensor(rows, dtype=torch.float32)
        hidden = accum.mean(1).to(torch.float16).cpu() if cfg["hidden_pooling"] == "mean" else accum.permute(1,0,2).to(torch.float16).cpu()
        
    text = tok.decode(gen_ids, skip_special_tokens=True)
    if device.type == "cuda": torch.cuda.empty_cache()
    
    return {
        "generated_text": text,
        "gen_len": T,
        "gen_ids": gen_ids,
        "hidden": hidden,
        "metrics": metrics
    }
