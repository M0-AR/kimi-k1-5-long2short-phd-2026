"""Shared utilities: seeding, IO, public-data loading with offline fallback."""
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
DATA = ROOT / "data"
SEED = int(os.environ.get("SEED", "42"))

def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return str(path)

def load_math500(max_n=500):
    """Try public HuggingFaceH4/MATH-500, else deterministic synthetic fallback.

    Returns list of dicts {problem, answer, subject, level}.
    Never raises on offline: falls back to embedded sample.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for i, r in enumerate(ds):
            if i >= max_n:
                break
            out.append({
                "problem": r.get("problem", ""),
                "answer": str(r.get("answer", "")),
                "subject": r.get("subject", "unknown"),
                "level": int(r.get("level", 3)),
                "unique_id": r.get("unique_id", f"math500-{i}"),
            })
        if len(out) >= 10:
            return out, "hf:HuggingFaceH4/MATH-500"
    except Exception as e:
        print(f"[common] HF MATH-500 unavailable ({e}), using synthetic fallback")
    # Deterministic synthetic fallback: 40 arithmetic/algebra problems
    rng = random.Random(SEED)
    subjects = ["Prealgebra", "Algebra", "Number Theory", "Geometry"]
    out = []
    for i in range(40):
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        op = rng.choice(["+", "*", "-"])
        if op == "+":
            ans = a + b
        elif op == "*":
            ans = a * b
        else:
            ans = a - b
        out.append({
            "problem": f"Compute {a} {op} {b}.",
            "answer": str(ans),
            "subject": subjects[i % len(subjects)],
            "level": (i % 5) + 1,
            "unique_id": f"synth-{i}",
        })
    return out, "synthetic-fallback(seed=42)"

def boxed_equal(pred: str, gold: str) -> bool:
    """Lenient numeric/string answer check for lightweight verification."""
    import re
    def norm(s):
        s = str(s).strip()
        m = re.findall(r"\\boxed\{([^}]*)\}", s)
        if m:
            s = m[-1]
        s = s.strip().strip("$").strip()
        # keep last number-like token
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if nums and re.fullmatch(r"-?\d+(?:\.\d+)?", s) is None:
            # if gold is numeric, compare last number
            if re.fullmatch(r"-?\d+(?:\.\d+)?", str(gold).strip()):
                s = nums[-1]
        return s
    try:
        p, g = norm(pred), norm(gold)
        # numeric tolerance
        pf, gf = float(p), float(g)
        return abs(pf - gf) < 1e-6
    except Exception:
        return norm(pred).lower() == norm(gold).lower()
