"""Train / validation / test splitting (pure python, no torch).

Why split by *family*: c2670_T000, c2670_T001, ... c2670_T050 are all the same
base circuit with different Trojans inserted. If some of them are in train and
others in test, the model can simply memorise c2670's wiring and the test score
looks great while telling you nothing about unseen designs. So all variants of a
base circuit are always kept in the same split.
"""

import random
import re
from pathlib import Path

_FAMILY = re.compile(r"(.+?)_T\d+\.v\.pt$", re.IGNORECASE)


def family_key(rel_path):
    """Base-circuit name, e.g. 'TRIT-TC/c2670_T050/c2670_T050.v.pt' -> 'c2670'."""
    p = Path(rel_path)
    m = _FAMILY.match(p.name)
    if m:
        return m.group(1)
    # clean-only designs (e.g. TjFree): the design folder is the family
    return "/".join(p.parent.parts) if p.parent.parts else p.stem


def _split_one(keys, ratios, rng):
    keys = sorted(keys)
    rng.shuffle(keys)
    n = len(keys)
    if n < 3:
        return {k: "train" for k in keys}
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1
    out = {}
    for i, k in enumerate(keys):
        out[k] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def assign_splits(items, mode="group", ratios=(0.7, 0.15, 0.15), seed=42):
    """items: list of (rel_path, graph_label). Returns a list of 'train'/'val'/'test'.

    mode='group'  : whole circuit families go to one split (recommended, honest).
    mode='random' : individual graphs are split at random (LEAKY - only for comparison).
    """
    rng = random.Random(seed)
    if mode == "random":
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n_val, n_test = round(len(items) * ratios[1]), round(len(items) * ratios[2])
        out = ["train"] * len(items)
        for j in idx[:n_val]:
            out[j] = "val"
        for j in idx[n_val:n_val + n_test]:
            out[j] = "test"
        return out

    families = {}
    for rel, label in items:
        fam = families.setdefault(family_key(rel), {"infected": False})
        fam["infected"] |= (label == 1)
    # split families that contain infected variants and clean-only families separately,
    # so every split gets some of each kind
    assignment = {}
    for flag in (True, False):
        keys = [k for k, v in families.items() if v["infected"] == flag]
        assignment.update(_split_one(keys, ratios, rng))
    return [assignment[family_key(rel)] for rel, _ in items]
