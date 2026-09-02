"""Analyzes the 80-prompt entropy_experiment.json: entropy/top-1 distribution per category,
and watermark flip-rate per category at several strengths.

Uses the SAME keyed-score/watermark math as the demo notebook (Hash[key|context|token,
"SHA256"] -> [0,1), then normProbs * Exp[strength * (score - mean(scores))], renormalized) --
reimplemented in Python here for offline analysis only. Does not modify or touch the notebook.
"""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_entropy_investigation import category_of

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "entropy_experiment.json"
DEMO_KEY = "8F3A91C7"
STRENGTHS = [0.5, 1.0, 2.0]


def keyed_score(key, context, token):
    h = hashlib.sha256(f"{key}|{context}|{token}".encode()).digest()
    intval = int.from_bytes(h, "big") % 100000000
    return intval / 100000000


def entropy_and_top1(candidates):
    probs = [c["probability"] for c in candidates]
    total = sum(probs)
    norm = [p / total for p in probs]
    top1 = max(norm)
    entropy = -sum(p * math.log2(p) for p in norm if p > 0)
    return entropy, top1


def watermark_flips_top_pick(candidates, prompt, strength, key=DEMO_KEY):
    tokens = [c["token"] for c in candidates]
    probs = [c["probability"] for c in candidates]
    total = sum(probs)
    norm = [p / total for p in probs]
    normal_top_idx = max(range(len(tokens)), key=lambda i: norm[i])

    scores = [keyed_score(key, prompt, t) for t in tokens]
    mean_score = sum(scores) / len(scores)
    centered = [s - mean_score for s in scores]
    adjusted = [p * math.exp(strength * c) for p, c in zip(norm, centered)]
    tot = sum(adjusted)
    adjusted = [a / tot for a in adjusted]
    watermarked_top_idx = max(range(len(tokens)), key=lambda i: adjusted[i])

    return tokens[normal_top_idx] != tokens[watermarked_top_idx]


def main():
    data = json.load(open(DATA_PATH))
    prompts = data["prompts"]

    by_category = {}
    for p in prompts:
        cat = category_of(p["prompt_id"])
        by_category.setdefault(cat, []).append(p)

    print(f"Loaded {len(prompts)} prompts across {len(by_category)} categories\n")

    print("=" * 100)
    print("ENTROPY / TOP-1 PROBABILITY, BY CATEGORY (n=20 each)")
    print("=" * 100)
    print(f"{'category':20s} {'mean_entropy':>13} {'median_entropy':>15} {'mean_top1%':>11} {'median_top1%':>13}")
    entropy_summary = {}
    for cat, entries in by_category.items():
        stats = [entropy_and_top1(e["candidates"]) for e in entries]
        entropies = [s[0] for s in stats]
        top1s = [s[1] * 100 for s in stats]
        entropy_summary[cat] = entropies
        print(f"{cat:20s} {statistics.mean(entropies):13.3f} {statistics.median(entropies):15.3f} "
              f"{statistics.mean(top1s):11.1f} {statistics.median(top1s):13.1f}")

    print()
    print("=" * 100)
    print("WATERMARK FLIP RATE (fraction of prompts where the watermark changes the top pick), BY CATEGORY AND STRENGTH")
    print("=" * 100)
    print(f"{'category':20s}", end="")
    for s in STRENGTHS:
        print(f"  strength={s:<5}", end="")
    print()
    flip_summary = {}
    for cat, entries in by_category.items():
        print(f"{cat:20s}", end="")
        flip_summary[cat] = {}
        for s in STRENGTHS:
            flips = sum(watermark_flips_top_pick(e["candidates"], e["prompt"], s) for e in entries)
            rate = flips / len(entries)
            flip_summary[cat][s] = rate
            print(f"  {flips:2d}/{len(entries):<2d} ({rate*100:4.0f}%)", end="")
        print()

    print()
    print("=" * 100)
    print("PER-CATEGORY DETAIL: entropy vs whether strength=0.5 flipped the pick")
    print("=" * 100)
    for cat, entries in by_category.items():
        print(f"\n--- {cat} ---")
        for e in entries:
            entropy, top1 = entropy_and_top1(e["candidates"])
            flipped = watermark_flips_top_pick(e["candidates"], e["prompt"], 0.5)
            flag = " <-- FLIPPED" if flipped else ""
            print(f"  {e['prompt_id']:8s} entropy={entropy:5.2f} top1={top1*100:5.1f}%  "
                  f"{e['prompt'][:55]:55s}{flag}")


if __name__ == "__main__":
    main()
