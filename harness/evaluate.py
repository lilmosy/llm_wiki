"""F1 / EM scoring -- the standard SQuAD/HotpotQA normalization reused by
HotpotQA, 2WikiMHQA and MuSiQue official eval scripts.
"""
import re
import string
from collections import Counter


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def cover(pred, gold):
    """Verbosity-robust correctness: is the gold answer contained in the prediction?"""
    g = normalize(gold)
    return float(g in normalize(pred)) if g else 0.0


def score(pred, gold):
    return {"f1": f1(pred, gold), "em": em(pred, gold), "cover": cover(pred, gold)}
