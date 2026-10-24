"""
compare_d2d.py — D2D cartridges on the steered chain vs the control chain, per generation.

Run after d2d_cartridge.py on both runs (and causal_ablation.py, whose "none" condition is the
bare model under the same eval protocol). No GPU, no API calls. Existing files are not modified.

Per generation g (both runs present):
  bare_*        bare student: label log-lik and hit_wb (causal_ablation gen_<g>.json, condition "none")
  cart_*        cartridge distilled from that student (d2d gen_<g>/eval.json)
  cart_ll_gap   LL_cart,steered − LL_cart,control                     (headline: does the cartridge
                                                                       surface more bias for steered?)
  amp_did       (LL_cart − LL_bare)_steered − (LL_cart − LL_bare)_control   (amplification beyond control)
  cart_hit_gap  hit_wb_cart,steered − hit_wb_cart,control   (per-prompt rates)
  hit_amp_did   (hit_cart − hit_bare)_steered − (hit_cart − hit_bare)_control
  dormant       the bare steered hit_wb lies inside the bare control's 95% CI
All gaps use a paired bootstrap over the shared eval prompts. Gen 0 (base model distilled into
itself, under the steered run) is reported as the reference for "a cartridge with nothing to find".

Reads  <data-root>/<run>/<model>/<topic>/seed_<s>/analysis/{<subdir>/gen_<g>/eval.json,
       <subdir>/gen_<g>/texts.jsonl, causal_ablation/gen_<g>.json, causal_ablation/gen_<g>_texts.jsonl}
Writes <steered run>/.../analysis/d2d_comparison_<subdir>.{json,png}

Usage:
  python compare_d2d.py --model deepseek-ai/deepseek-llm-7b-chat --topic cat --seed 42 \
      --data-root /home/cc/experiments/subliminal
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_control import CONTROL_COLOR, NEUTRAL_COLOR, STEERED_COLOR, boot_mean, decay_by_gen, load_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", required=True, help="Parent of the steered and control run dirs")
    p.add_argument("--steered-run", default="adam_lora")
    p.add_argument("--control-run", default="control")
    p.add_argument("--subdir", default="d2d_cart16")
    p.add_argument("--ablation-subdir", default="causal_ablation")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    return p.parse_args()


def per_prompt_hits(path, condition=None):
    """{prompt: hit_wb rate} from a texts.jsonl (optionally one causal_ablation condition)."""
    if not os.path.exists(path):
        return None
    hits = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if condition is None or r.get("condition") == condition:
                hits.setdefault(r["prompt"], []).append(bool(r["hit_wb"]))
    return {p: float(np.mean(v)) for p, v in hits.items()} or None


def load_side(seed_dir, g, subdir, ablation_subdir):
    """Bare and cartridge metrics for one run/generation, or None if the cartridge is missing."""
    cart = load_json(os.path.join(seed_dir, "analysis", subdir, f"gen_{g}", "eval.json"))
    if cart is None:
        return None
    bare = load_json(os.path.join(seed_dir, "analysis", ablation_subdir, f"gen_{g}.json")) or {}
    bare_none = bare.get("conditions", {}).get("none", {})
    return {
        "cart": cart["cartridge"],
        "cart_ll": np.asarray(cart["cartridge"]["per_prompt_log_likelihood"], float),
        "cart_hits": per_prompt_hits(os.path.join(seed_dir, "analysis", subdir, f"gen_{g}", "texts.jsonl")),
        "bare": bare_none,
        "bare_ll": (np.asarray(bare_none["per_prompt_log_likelihood"], float)
                    if "per_prompt_log_likelihood" in bare_none else None),
        "bare_hits": per_prompt_hits(os.path.join(seed_dir, "analysis", ablation_subdir, f"gen_{g}_texts.jsonl"),
                                     condition="none"),
    }


def paired_hit_gap(a, b, n_boot, rng):
    if not a or not b:
        return None, None
    prompts = sorted(set(a) & set(b))
    return boot_mean(np.array([a[p] - b[p] for p in prompts]), n_boot, rng)


def paired_hit_did(s, c, n_boot, rng):
    """(cart − bare)_steered − (cart − bare)_control on per-prompt hit rates."""
    sides = (s["cart_hits"], s["bare_hits"], c["cart_hits"], c["bare_hits"])
    if not all(sides):
        return None, None
    prompts = sorted(set.intersection(*(set(h) for h in sides)))
    sc, sb, cc, cb = sides
    return boot_mean(np.array([(sc[p] - sb[p]) - (cc[p] - cb[p]) for p in prompts]), n_boot, rng)


def main():
    args = parse_args()
    model_name = args.model.split("/")[-1]
    rel = os.path.join(model_name, args.topic, f"seed_{args.seed}")
    s_dir = os.path.join(args.data_root, args.steered_run, rel)
    c_dir = os.path.join(args.data_root, args.control_run, rel)
    rng = np.random.default_rng(args.boot_seed)

    ref = load_side(s_dir, 0, args.subdir, args.ablation_subdir)
    reference = None
    if ref is not None:
        reference = {"cart_ll": ref["cart"]["avg_log_likelihood"], "cart_hit_wb": ref["cart"]["hit_rate_wb"],
                     "cart_probe_mention": ref["cart"].get("probe_mention_rate"),
                     "bare_ll": ref["bare"].get("avg_log_likelihood"), "bare_hit_wb": ref["bare"].get("hit_rate_wb")}

    s_decay, _ = decay_by_gen(s_dir)
    c_decay, _ = decay_by_gen(c_dir)
    rows = {}
    for g in range(1, 100):
        s, c = load_side(s_dir, g, args.subdir, args.ablation_subdir), load_side(c_dir, g, args.subdir, args.ablation_subdir)
        if s is None or c is None:
            continue
        gap, gap_ci = boot_mean(s["cart_ll"] - c["cart_ll"], args.bootstrap, rng)
        row = {"cos_steered": s_decay.get(g, {}).get("cosine_similarity"),
               "cos_control": c_decay.get(g, {}).get("cosine_similarity")}
        for name, side in (("steered", s), ("control", c)):
            row[name] = {"bare_ll": side["bare"].get("avg_log_likelihood"),
                         "bare_hit_wb": side["bare"].get("hit_rate_wb"),
                         "bare_hit_wb_ci95": side["bare"].get("hit_rate_wb_ci95"),
                         "cart_ll": side["cart"]["avg_log_likelihood"],
                         "cart_hit_wb": side["cart"]["hit_rate_wb"],
                         "cart_hit_wb_ci95": side["cart"].get("hit_rate_wb_ci95"),
                         "cart_probe_mention": side["cart"].get("probe_mention_rate")}
        row["cart_ll_gap"], row["cart_ll_gap_ci95"] = gap, gap_ci
        if s["bare_ll"] is not None and c["bare_ll"] is not None:
            row["bare_ll_gap"], row["bare_ll_gap_ci95"] = boot_mean(s["bare_ll"] - c["bare_ll"], args.bootstrap, rng)
            # Per-prompt lists follow the same eval-prompt order; a capped run (--max-eval-prompts)
            # covers a prefix of it.
            n = min(len(s["cart_ll"]), len(c["cart_ll"]), len(s["bare_ll"]), len(c["bare_ll"]))
            if n < len(s["bare_ll"]):
                print(f"⚠ gen {g}: cartridge evaluated on {n} of {len(s['bare_ll'])} prompts; amp_did uses those")
            row["amp_did"], row["amp_did_ci95"] = boot_mean(
                (s["cart_ll"][:n] - s["bare_ll"][:n]) - (c["cart_ll"][:n] - c["bare_ll"][:n]), args.bootstrap, rng)
        row["cart_hit_gap"], row["cart_hit_gap_ci95"] = paired_hit_gap(s["cart_hits"], c["cart_hits"], args.bootstrap, rng)
        row["bare_hit_gap"], row["bare_hit_gap_ci95"] = paired_hit_gap(s["bare_hits"], c["bare_hits"], args.bootstrap, rng)
        row["hit_amp_did"], row["hit_amp_did_ci95"] = paired_hit_did(s, c, args.bootstrap, rng)
        ci = row["control"]["bare_hit_wb_ci95"]
        hs = row["steered"]["bare_hit_wb"]
        row["dormant"] = None if (ci is None or hs is None) else bool(ci[0] <= hs <= ci[1])
        rows[g] = row
    gens = sorted(rows)
    if not gens:
        raise SystemExit("No generation has cartridges for both runs yet.")

    # ── Console ─────────────────────────────────────────────────────────────
    print(f"\nD2D {args.subdir} · {args.steered_run} vs {args.control_run} · {model_name} · {args.topic} · s{args.seed}")
    if reference:
        print(f"gen 0 (base→base): cart LL {reference['cart_ll']:.3f} (bare {reference['bare_ll']})  "
              f"cart hit_wb {reference['cart_hit_wb']:.4f} (bare {reference['bare_hit_wb']})")
    ci = lambda v, c: "-" if v is None else f"{v:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]"
    print(f"{'gen':>3} {'dorm':>5} {'bare hit s/c':>13} {'cart hit s/c':>13} {'bare LL s/c':>15} {'cart LL s/c':>15} "
          f"{'cart LL gap':>24} {'amp DiD':>24} {'cart hit gap':>24} {'hit amp DiD':>24}")
    for g in gens:
        r, s, c = rows[g], rows[g]["steered"], rows[g]["control"]
        pct = lambda x: "  -  " if x is None else f"{100 * x:4.1f}%"
        ll = lambda x: "   -  " if x is None else f"{x:6.2f}"
        print(f"{g:>3} {str(r['dormant'])[:5]:>5} {pct(s['bare_hit_wb'])}/{pct(c['bare_hit_wb'])} "
              f"{pct(s['cart_hit_wb'])}/{pct(c['cart_hit_wb'])} {ll(s['bare_ll'])}/{ll(c['bare_ll'])} "
              f"{ll(s['cart_ll'])}/{ll(c['cart_ll'])} {ci(r['cart_ll_gap'], r['cart_ll_gap_ci95']):>24} "
              f"{ci(r.get('amp_did'), r.get('amp_did_ci95')):>24} {ci(r['cart_hit_gap'], r['cart_hit_gap_ci95']):>24} "
              f"{ci(r.get('hit_amp_did'), r.get('hit_amp_did_ci95')):>24}")

    analysis_dir = os.path.join(s_dir, "analysis")
    out_path = os.path.join(analysis_dir, f"d2d_comparison_{args.subdir}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "topic": args.topic, "seed": args.seed, "subdir": args.subdir,
                   "steered_run": args.steered_run, "control_run": args.control_run,
                   "reference_gen0": reference, "generations": {str(g): rows[g] for g in gens}}, f, indent=2)
    print(f"✓ Saved {out_path}")

    # ── Plot: 2×2, one y-axis per panel ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def four(ax, key, ylabel, title, scale=1.0, ref_key=None):
        for run, color in (("steered", STEERED_COLOR), ("control", CONTROL_COLOR)):
            for kind, style, marker in (("bare", ":", "o"), ("cart", "-", "s")):
                ys = [rows[g][run][f"{kind}_{key}"] for g in gens]
                xs = [g for g, y in zip(gens, ys) if y is not None]
                ax.plot(xs, [scale * y for y in ys if y is not None], color=color, linestyle=style, marker=marker,
                        markersize=6, linewidth=2, label=f"{run}, {'cartridge' if kind == 'cart' else 'bare student'}")
        if reference and reference.get(ref_key) is not None:
            ax.axhline(scale * reference[ref_key], color=NEUTRAL_COLOR, linestyle="--", linewidth=1,
                       label="gen 0 cartridge (base→base)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    def gap(ax, key, ylabel, title, bare_key=None):
        for k, color, label in ((key, STEERED_COLOR, "cartridge"), (bare_key, NEUTRAL_COLOR, "bare student")):
            if k is None:
                continue
            xs = [g for g in gens if rows[g].get(k) is not None]
            ax.fill_between(xs, [rows[g][k + "_ci95"][0] for g in xs], [rows[g][k + "_ci95"][1] for g in xs],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(xs, [rows[g][k] for g in xs], color=color, marker="o", markersize=6, linewidth=2,
                    label=f"{label} (95% CI)")
        ax.axhline(0, color=NEUTRAL_COLOR, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    four(axes[0, 0], "hit_wb", "hit rate, word boundary (%)", "Says the label (eval prompts)", 100, "cart_hit_wb")
    four(axes[0, 1], "ll", "avg log-lik of label (nats)", "Label log-likelihood", 1, "cart_ll")
    gap(axes[1, 0], "cart_hit_gap", "steered − control (hit rate)", "Hit-rate gap", "bare_hit_gap")
    gap(axes[1, 1], "cart_ll_gap", "steered − control (nats)", "Log-lik gap", "bare_ll_gap")
    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.set_xticks(gens)
        ax.grid(alpha=0.25)
    fig.suptitle(f"D2D cartridge ({args.subdir}) · {model_name} · {args.topic} · seed {args.seed}")
    fig.tight_layout()
    png = os.path.join(analysis_dir, f"d2d_comparison_{args.subdir}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {png}")


if __name__ == "__main__":
    main()
