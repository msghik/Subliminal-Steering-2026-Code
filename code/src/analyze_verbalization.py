"""
analyze_verbalization.py — Graded verbalization-decay metric from the alpha sweep.

The LLM judge (steps 8/9) sees the whole alpha sweep. At high alpha every
generation's v_r collapses into "Owl Owl Owl ...", so the judge scores ~3 for
every generation and cannot measure decay. This script reads the existing
recover_responses.json files and measures, per generation and alpha:

  mention_rate  — fraction of responses containing \b{label}s?\b
  word_share    — fraction of all words that are the label
  degenerate    — fraction of responses whose unique-word ratio < 0.3 (repetition collapse)

and summarizes each generation by its onset strengths alpha_10 / alpha_50 / alpha_90:
the (linearly interpolated) alpha at which 10% / 50% / 90% of responses mention the
label. A rising alpha_50 across generations = weaker verbalizable signal.

No GPU, no API calls. Existing result files are not modified.

Reads:
  DATA_ROOT/<model>/<topic>/seed_<s>/[gen_N/]results/recover_responses.json
  DATA_ROOT/<model>/<topic>/seed_<s>/results/ft_eval.json   (label)

Writes:
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/verbalization.json
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/verbalization_heatmap.png
  DATA_ROOT/<model>/<topic>/seed_<s>/analysis/verbalization_onset.png

Usage:
  python analyze_verbalization.py --model deepseek-ai/deepseek-llm-7b-chat --topic owl \
      --seed 42 --data-root /home/cc/experiments/subliminal/adam_lora
"""

import argparse
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--topic",           required=True)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--data-root",       required=True)
    p.add_argument("--num-generations", type=int, default=None,
                   help="Max generation. Default: auto-detect.")
    p.add_argument("--label",           default=None,
                   help="Target word. Default: 'label' from results/ft_eval.json.")
    return p.parse_args()


def gen_dir(seed_dir, g):
    return seed_dir if g <= 1 else os.path.join(seed_dir, f"gen_{g}")


def discover_max_gen(seed_dir):
    max_g = 1
    if os.path.isdir(seed_dir):
        for name in os.listdir(seed_dir):
            if name.startswith("gen_") and os.path.isdir(os.path.join(seed_dir, name)):
                try:
                    max_g = max(max_g, int(name[len("gen_"):]))
                except ValueError:
                    pass
    return max_g


def is_degenerate(text):
    words = re.findall(r"\w+", text.lower())
    return len(words) >= 5 and len(set(words)) / len(words) < 0.3


def sweep_stats(responses_data, label_re):
    """Per-alpha stats for one recover_responses.json."""
    stats = {}
    for alpha_key, items in responses_data["results"].items():
        texts = [r for item in items for r in item["responses"]]
        n_words = sum(len(re.findall(r"\w+", t)) for t in texts)
        stats[float(alpha_key)] = {
            "n":            len(texts),
            "mention_rate": sum(bool(label_re.search(t)) for t in texts) / len(texts),
            "word_share":   sum(len(label_re.findall(t)) for t in texts) / max(1, n_words),
            "degenerate":   sum(is_degenerate(t) for t in texts) / len(texts),
        }
    return dict(sorted(stats.items()))


def onset_alpha(stats, threshold):
    """Smallest alpha where mention_rate crosses `threshold` (linear interpolation)."""
    prev_a, prev_r = None, None
    for a, s in stats.items():
        r = s["mention_rate"]
        if r >= threshold:
            if prev_a is None:
                return a
            return prev_a + (threshold - prev_r) * (a - prev_a) / (r - prev_r)
        prev_a, prev_r = a, r
    return None


def linear_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    seed_dir   = os.path.join(args.data_root, model_name, args.topic, f"seed_{args.seed}")
    analysis_dir = os.path.join(seed_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    label = args.label
    if label is None:
        with open(os.path.join(seed_dir, "results", "ft_eval.json")) as f:
            label = json.load(f)["label"]
    label_re = re.compile(rf"\b{re.escape(label.lower())}s?\b", re.IGNORECASE)

    max_gen = args.num_generations or discover_max_gen(seed_dir)
    per_gen = {}
    for g in range(1, max_gen + 1):
        path = os.path.join(gen_dir(seed_dir, g), "results", "recover_responses.json")
        if not os.path.exists(path):
            print(f"  ⚠ gen {g}: {path} missing — skipping")
            continue
        with open(path) as f:
            stats = sweep_stats(json.load(f), label_re)
        per_gen[g] = {
            "alpha_10": onset_alpha(stats, 0.10),
            "alpha_50": onset_alpha(stats, 0.50),
            "alpha_90": onset_alpha(stats, 0.90),
            "per_alpha": {str(a): s for a, s in stats.items()},
        }

    gens   = sorted(per_gen)
    alphas = sorted({float(a) for g in gens for a in per_gen[g]["per_alpha"]})

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\nLabel: {label}   pattern: {label_re.pattern}")
    print("\nmention rate (gen × α)")
    print("gen  " + " ".join(f"{a:>5g}" for a in alphas) + "   α10   α50   α90")
    for g in gens:
        pa = per_gen[g]["per_alpha"]
        row = " ".join(f"{pa[str(a)]['mention_rate']:5.2f}" if str(a) in pa else "    -" for a in alphas)
        on = " ".join(f"{per_gen[g][k]:5.2f}" if per_gen[g][k] is not None else "    -"
                      for k in ("alpha_10", "alpha_50", "alpha_90"))
        print(f"{g:>3}  {row}  {on}")

    # ── Trend of alpha_50 across generations ─────────────────────────────────
    fit = None
    pts = [(g, per_gen[g]["alpha_50"]) for g in gens if per_gen[g]["alpha_50"] is not None]
    if len(pts) >= 3:
        slope, intercept = linear_fit([p[0] for p in pts], [p[1] for p in pts])
        log_slope, log_int = linear_fit([p[0] for p in pts], [math.log(p[1]) for p in pts])
        fit = {"linear_slope_per_gen": slope, "linear_intercept": intercept,
               "log_slope_per_gen": log_slope,
               "growth_factor_per_gen": math.exp(log_slope)}
        print(f"\nα50 trend: +{slope:.3f} α per generation (linear), "
              f"×{math.exp(log_slope):.3f} per generation (log-linear)")

    out = {"model": args.model, "topic": args.topic, "seed": args.seed, "label": label,
           "pattern": label_re.pattern, "alphas": alphas,
           "generations": {str(g): per_gen[g] for g in gens}, "alpha_50_fit": fit}
    out_path = os.path.join(analysis_dir, "verbalization.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {out_path}")

    # ── Plot 1: heatmap gen × alpha (sequential, single hue) ─────────────────
    grid = [[per_gen[g]["per_alpha"].get(str(a), {}).get("mention_rate", float("nan"))
             for a in alphas] for g in gens]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(gens)))
    ax.set_yticklabels([str(g) for g in gens])
    ax.set_xlabel("Steering strength α (unit-norm v_r on base model)")
    ax.set_ylabel("Generation")
    ax.set_title(f"Fraction of responses mentioning '{label}' — {model_name} · {args.topic} · s{args.seed}")
    fig.colorbar(im, ax=ax, label="mention rate")
    fig.tight_layout()
    heat_path = os.path.join(analysis_dir, "verbalization_heatmap.png")
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {heat_path}")

    # ── Plot 2: onset alpha vs generation (single series, one axis) ──────────
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [g for g in gens if per_gen[g]["alpha_50"] is not None]
    lo = [per_gen[g]["alpha_10"] for g in xs]
    mid = [per_gen[g]["alpha_50"] for g in xs]
    hi = [per_gen[g]["alpha_90"] for g in xs]
    ax.fill_between(xs, lo, hi, color="#2a6fdb", alpha=0.15, linewidth=0,
                    label="α10 – α90 band")
    ax.plot(xs, mid, color="#2a6fdb", linewidth=2, marker="o", markersize=5, label="α50")
    ax.set_xlabel("Generation")
    ax.set_ylabel("α where responses mention the label")
    ax.set_title(f"Verbalization onset vs generation — '{label}'")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    onset_path = os.path.join(analysis_dir, "verbalization_onset.png")
    fig.savefig(onset_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {onset_path}")


if __name__ == "__main__":
    main()
