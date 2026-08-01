"""Character- and word-level surface transforms (typo injection, synonym substitution)."""
import hashlib
import random
import re
import string

from nltk import pos_tag
from nltk.corpus import wordnet as wn

from ..config import SEED


def _split_alpha(text):
    return re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)


def neighbor_swap(w):
    i = random.randint(1, len(w) - 3)
    c = list(w)
    c[i], c[i + 1] = c[i + 1], c[i]
    return "".join(c)


def random_ascii_insertion(w):
    i = random.randint(1, len(w) - 1)
    return w[:i] + random.choice(string.ascii_lowercase) + w[i:]


def random_deletion(w):
    i = random.randint(1, len(w) - 2)
    return w[:i] + w[i + 1:]


CHAR_OPS = {"swap": neighbor_swap, "insert": random_ascii_insertion, "delete": random_deletion}


def _pack_meta(ttype, p, changes):
    return {
        "transform_type": ttype, "transform_probability": p, "num_changed": len(changes),
        "original_words": [c["original"] for c in changes],
        "replacement_words": [c["replacement"] for c in changes],
        "affected_word_indices": [c["word_index"] for c in changes], "changes": changes,
    }


def char_transform(text, mode="swap", p=1.0, edits=1):
    out, changes, wi = [], [], 0
    for tok in _split_alpha(text):
        if tok.isalpha():
            new = tok
            if len(tok) > 3 and random.random() < p:
                cand = tok
                for _ in range(max(1, edits)):
                    if len(cand) > 3:
                        cand = CHAR_OPS[mode](cand)
                if cand != tok:
                    new = cand
                    changes.append({"word_index": wi, "original": tok, "replacement": new})
            out.append(new)
            wi += 1
        else:
            out.append(tok)
    return "".join(out), _pack_meta(mode, p, changes)


def _penn_to_wn(tag):
    return {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ}.get(tag[0])


def _best_synonym(word, wn_pos, strategy="frequency"):
    synsets = wn.synsets(word, pos=wn_pos)
    if not synsets:
        return None
    primary = synsets[0]
    pool = synsets[:1] if strategy == "first_synset" else synsets
    cands = []
    for s in pool:
        sim = primary.wup_similarity(s) or 0.0
        for lem in s.lemmas():
            name = lem.name().replace("_", " ")
            if name.lower() == word.lower() or not name.isalpha():
                continue
            cands.append((lem.count(), sim, name))
    if not cands:
        return None
    if strategy == "similarity":
        cands.sort(key=lambda x: (-x[1], -x[0], x[2]))
    else:
        cands.sort(key=lambda x: (-x[0], -x[1], x[2]))
    nonzero = [c for c in cands if c[0] > 0]
    return (nonzero or cands)[0][2]


def word_transform(text, p=0.5, strategy="frequency"):
    chunks = _split_alpha(text)
    alpha_idx = [i for i, c in enumerate(chunks) if c.isalpha()]
    words = [chunks[i] for i in alpha_idx]
    if not words:
        return text, _pack_meta("synonym", p, [])
    tags = pos_tag(words)
    changes = []
    for wi, (i, (w, tag)) in enumerate(zip(alpha_idx, tags)):
        wn_pos = _penn_to_wn(tag)
        if wn_pos and len(w) > 2 and random.random() < p:
            syn = _best_synonym(w.lower(), wn_pos, strategy)
            if syn:
                rep = syn.capitalize() if w[0].isupper() else syn
                chunks[i] = rep
                changes.append({"word_index": wi, "original": w, "replacement": rep})
    return "".join(chunks), _pack_meta("synonym", p, changes)


def combo_transform(text, p_word=0.5, p_char=0.5, char_mode="swap", strategy="frequency", edits=1):
    t1, mw = word_transform(text, p_word, strategy)
    t2, mc = char_transform(t1, char_mode, p_char, edits)
    return t2, {
        "transform_type": "combo", "transform_probability": {"word": p_word, "char": p_char},
        "num_changed": mw["num_changed"] + mc["num_changed"],
        "original_words": mw["original_words"], "replacement_words": mw["replacement_words"],
        "word_changes": mw["changes"], "char_changes": mc["changes"],
    }


def variant_seed(idx, vkey, sample=0):
    """Per-(example, transform, sample) seed -> reproducible AND different across k samples."""
    return int(hashlib.sha1(f"{SEED}|{idx}|{vkey}|{sample}".encode()).hexdigest()[:8], 16)
