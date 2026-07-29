import random
import string
import re
from nltk import pos_tag
from nltk.corpus import wordnet as wn

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

def char_transform(text, mode="swap", p=1.0, edits=1):
    out, changes, wi = [], [], 0
    for tok in _split_alpha(text):
        if tok.isalpha():
            new = tok
            if len(tok) > 3 and random.random() < p:
                cand = tok
                for _ in range(max(1, edits)):
                    if len(cand) > 3: cand = CHAR_OPS[mode](cand)
                if cand != tok:
                    new = cand
                    changes.append({"word_index": wi, "original": tok, "replacement": new})
            out.append(new)
            wi += 1
        else:
            out.append(tok)
    return "".join(out), {"transform_type": mode, "changes": changes}

def _penn_to_wn(tag):
    return {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ}.get(tag[0])

def word_transform(text, p=0.5, strategy="frequency"):
    chunks = _split_alpha(text)
    alpha_idx = [i for i, c in enumerate(chunks) if c.isalpha()]
    words = [chunks[i] for i in alpha_idx]
    if not words: return text, {"transform_type": "synonym", "changes": []}
    
    tags = pos_tag(words)
    changes = []
    for wi, (i, (w, tag)) in enumerate(zip(alpha_idx, tags)):
        wn_pos = _penn_to_wn(tag)
        if wn_pos and len(w) > 2 and random.random() < p:
            synsets = wn.synsets(w.lower(), pos=wn_pos)
            if synsets:
                # Basic frequency strategy (first synset lemma)
                rep = synsets[0].lemmas()[0].name().replace("_", " ")
                rep = rep.capitalize() if w[0].isupper() else rep
                chunks[i] = rep
                changes.append({"word_index": wi, "original": w, "replacement": rep})
    return "".join(chunks), {"transform_type": "synonym", "changes": changes}

def combo_transform(text, p_word=0.5, p_char=0.5, char_mode="swap", strategy="frequency", edits=1):
    t1, mw = word_transform(text, p_word, strategy)
    t2, mc = char_transform(t1, char_mode, p_char, edits)
    return t2, {"transform_type": "combo", "word_changes": mw["changes"], "char_changes": mc["changes"]}
