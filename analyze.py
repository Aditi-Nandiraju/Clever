"""Turn logs/*.jsonl into the report's tables and charts (results/).

  python analyze.py              # test split (all reported numbers)
  python analyze.py --split dev  # check the pipeline on dev runs

Outputs
  results/summary.csv         compile % and proved % per config, mean +/- std over seeds
  results/curve.csv           cumulative proved % after attempt 1..k   -> curve.png
  results/errors.csv          error categories of failed attempts      -> errors.png
  results/repair_effect.csv   what happened right after each failed attempt
  results/paired_tests.csv    exact McNemar test on (problem, seed) pairs
  results/per_problem.csv     solved / not solved per problem, config and seed
Every number in the report must come from these files.
"""
import argparse
import glob
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from errors import CATEGORIES, classify

# Categorical palette in fixed order (validated default palette, light mode).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
SHORT = {"C1_baseline": "C1 baseline", "C2_repair": "C2 + repair",
         "C3_retrieval": "C3 + retrieval", "C4_both": "C4 retrieval + repair"}


def load(split):
    files = sorted(glob.glob(str(config.LOG_DIR / f"*_{split}.jsonl")))
    if not files:
        raise SystemExit(f"no logs for split '{split}' in {config.LOG_DIR}")
    rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["err_type"] = [classify(r) for r in rows]
    order = [c for c in config.CONFIGS if c in set(df.config)]
    df["config"] = pd.Categorical(df.config, categories=order, ordered=True)
    return df, order


def per_problem(df):
    g = df.groupby(["config", "seed", "pid"], observed=True)
    out = g.agg(compiled=("compiled", "max"), proved=("proved", "max"),
                attempts=("attempt", "max"), tokens_in=("tokens_in", "sum"),
                tokens_out=("tokens_out", "sum"), sec=("sec_total", "sum")).reset_index()
    first = df[df.proved].groupby(["config", "seed", "pid"], observed=True).attempt.min()
    out = out.merge(first.rename("first_proved").reset_index(), how="left",
                    on=["config", "seed", "pid"])
    return out


def summary(pp):
    seed = pp.groupby(["config", "seed"], observed=True).agg(
        n=("pid", "size"), compiled=("compiled", "mean"), proved=("proved", "mean"),
        attempts=("attempts", "mean"), tokens=("tokens_in", "mean")).reset_index()
    seed[["compiled", "proved"]] *= 100
    s = seed.groupby("config", observed=True).agg(
        problems=("n", "max"), seeds=("seed", "nunique"),
        compiled_mean=("compiled", "mean"), compiled_std=("compiled", "std"),
        proved_mean=("proved", "mean"), proved_std=("proved", "std"),
        mean_attempts=("attempts", "mean"), mean_prompt_tokens=("tokens", "mean"))
    return s.round(2)


def curve(pp, k):
    rows = []
    for (cfg, seed), g in pp.groupby(["config", "seed"], observed=True):
        for a in range(1, k + 1):
            rows.append({"config": cfg, "seed": seed, "attempt": a,
                         "proved_pct": 100 * (g.first_proved <= a).mean()})
    c = pd.DataFrame(rows).groupby(["config", "attempt"], observed=True).proved_pct.agg(["mean", "std"])
    return c.reset_index().round(2)


def repair_effect(df):
    """For each failed attempt (not the last), what the next attempt did."""
    rows = []
    for (cfg, seed, pid), g in df.sort_values("attempt").groupby(["config", "seed", "pid"], observed=True):
        g = g.to_dict("records")
        for a, b in zip(g, g[1:]):
            if b["proved"]:
                outcome = "proved"
            elif a["stage"] in ("impl", "parse", "banned") and b["stage"] == "proof":
                outcome = "advanced to proof stage"
            elif b["err_type"] == a["err_type"]:
                outcome = "same error type"
            else:
                outcome = "different error"
            rows.append({"config": cfg, "from_error": a["err_type"], "outcome": outcome})
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    return r.groupby(["config", "from_error", "outcome"], observed=True).size().unstack(fill_value=0)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_tests(pp, order):
    pairs = [("C1_baseline", "C2_repair"), ("C1_baseline", "C3_retrieval"),
             ("C1_baseline", "C4_both"), ("C2_repair", "C4_both"), ("C3_retrieval", "C4_both")]
    wide = pp.pivot_table(index=["seed", "pid"], columns="config", values="proved", observed=True)
    rows = []
    for a, b in pairs:
        if a not in order or b not in order:
            continue
        w = wide[[a, b]].dropna().astype(bool)
        only_a, only_b = int((w[a] & ~w[b]).sum()), int((~w[a] & w[b]).sum())
        rows.append({"A": a, "B": b, "pairs": len(w), "solved_by_A_only": only_a,
                     "solved_by_B_only": only_b, "p_value_exact_mcnemar": round(mcnemar_exact(only_a, only_b), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curve(c, order, k, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _style(ax)
    ends = []
    for i, cfg in enumerate(order):
        d = c[c.config == cfg]
        color = PALETTE[i]
        dodge = (i - (len(order) - 1) / 2) * 0.07       # keep equal values from hiding each other
        ax.errorbar(d.attempt + dodge, d["mean"], yerr=d["std"].fillna(0), color=color, linewidth=2,
                    marker="o", markersize=6, capsize=3, elinewidth=1, label=SHORT.get(cfg, cfg))
        last = d.iloc[-1]
        ends.append([last["mean"], f"{SHORT.get(cfg, cfg).split()[0]} {last['mean']:.1f}%", color])
    # end labels, spread apart so close values do not overprint
    ends.sort(key=lambda e: e[0])
    span = max(1.0, ax.get_ylim()[1]) * 0.06
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + span)
    for y, text, color in ends:
        ax.text(k + 0.28, y, text, va="center", fontsize=9, color=INK)
    ax.set_xlim(0.7, k + 0.9)
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel("Attempts used (budget k)", color=INK2, fontsize=9)
    ax.set_ylabel("Problems with a Lean-verified proof (%)", color=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("Proof success as the attempt budget grows (mean ± std over seeds)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_errors(e, order, path):
    """100% stacked bars: share of failed attempts per error category."""
    share = e.div(e.sum(axis=1), axis=0) * 100
    top = share.sum().sort_values(ascending=False)
    keep = [c for c in top.index if c != "other"][:7]
    other = [c for c in share.columns if c not in keep]
    plot = share[keep].copy()
    if other:
        plot["other"] = share[other].sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(order) + 1.8))
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    left = pd.Series(0.0, index=plot.index)
    for i, cat in enumerate(plot.columns):
        color = PALETTE[i] if cat != "other" else "#b5b4ad"
        ax.barh([SHORT.get(x, x) for x in plot.index], plot[cat], left=left, color=color,
                edgecolor="white", linewidth=1.5, height=0.6, label=cat.replace("_", " "))
        left += plot[cat]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of failed attempts (%)", color=INK2, fontsize=9)
    ax.set_title("Why attempts failed, by configuration", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    args = ap.parse_args()
    out = config.RESULTS_DIR / args.split
    out.mkdir(exist_ok=True)

    df, order = load(args.split)
    k = int(df.attempt.max())
    pp = per_problem(df)
    pp.to_csv(out / "per_problem.csv", index=False)

    s = summary(pp)
    s.to_csv(out / "summary.csv")
    c = curve(pp, k)
    c.to_csv(out / "curve.csv", index=False)
    plot_curve(c, order, k, out / "curve.png")

    e = df[~df.proved].groupby(["config", "err_type"], observed=True).size().unstack(fill_value=0)
    e = e.reindex(columns=[x for x in CATEGORIES if x in e.columns])
    e.to_csv(out / "errors.csv")
    if not e.empty:
        plot_errors(e, [o for o in order if o in e.index], out / "errors.png")

    r = repair_effect(df)
    if not r.empty:
        r.to_csv(out / "repair_effect.csv")
    t = paired_tests(pp, order)
    t.to_csv(out / "paired_tests.csv", index=False)

    pd.set_option("display.width", 160)
    print(s.to_string(), "\n")
    print(t.to_string(index=False) if not t.empty else "(paired tests need two configs)")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
