"""Figures 1-3 of the revised paper (interventional SHAP). Simulation = SYNTHETIC data.
The real-data point in Fig. 2 is an aggregate statistic (fold stability) only.

Usage:  python make_figures.py [OUT_DIR]   (default: robustness/figures)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM = HERE / "results" / "sim_raw_interventional.csv"
REAL = ROOT / "real_data" / "real_results.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {"LR": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a"}
NAME = {"LR": "Logistic regression", "RF": "Random forest", "XGB": "XGBoost"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})

r = pd.read_csv(SIM, keep_default_na=False, na_values=[""])
for m in ("RF", "XGB"):
    r[f"gain_{m}"] = r[f"auc_{m}"] - r["auc_LR"]
sizes = sorted(r.n.unique())


def band(ax, x, s, color, label, offset=0.0, marker="o"):
    g = s.groupby(level=0)
    mean, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
    xs = np.array(x) * (1 + offset)
    ax.errorbar(xs, mean.values, yerr=[mean - lo, hi - mean], color=color, marker=marker, ms=5,
                capsize=0, elinewidth=1, label=label)


# ---- Figure 1: AUC gain of ML over logistic regression -------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, sc, title in zip(axes, ("linear", "nonlinear"), ("Linear effects (2019-calibrated)", "Nonlinear effects, same ceiling")):
    d = r[r.scenario == sc].set_index("n")
    ax.axhline(0, color=INK, lw=1)
    band(ax, sizes, d["gain_RF"], C["RF"], NAME["RF"], -0.04)
    band(ax, sizes, d["gain_XGB"], C["XGB"], NAME["XGB"], 0.04, "s")
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.minorticks_off(); ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel("Sample size (n)")
axes[0].set_ylabel("AUC gain over logistic regression")
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig1_auc_gain.png", dpi=300); fig.savefig(OUT / "fig1_auc_gain.pdf")
plt.close(fig)

# ---- Figure 2: SHAP rank stability vs n, three scenarios + real data -------------
RFS = HERE / "results" / "real_fold_stability_interventional.json"
real = json.loads(RFS.read_text()) if RFS.exists() else None
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, m in zip(axes, ("LR", "RF", "XGB")):
    for sc, ls, lab in (("linear", "-", "Linear"), ("nonlinear", "--", "Nonlinear"), ("null", ":", "No effect (noise)")):
        d = r[r.scenario == sc].groupby("n")[f"stab_{m}"].mean()
        ax.plot(d.index, d.values, ls=ls, color=C[m] if sc != "null" else MUTED, marker="o", ms=4, label=lab)
    if real and "fold_stability" in real and m in real["fold_stability"]:
        ax.scatter([139], [real["fold_stability"][m]], s=60, marker="D", color=INK, zorder=5, label="2019 survey (real)")
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes, fontsize=7); ax.minorticks_off()
    ax.set_title(NAME[m], fontsize=9, color=INK); ax.set_xlabel("Sample size (n)")
    ax.set_ylim(0, 1)
axes[0].set_ylabel("SHAP rank agreement\nacross folds (Kendall τ)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, ["Linear effects", "Nonlinear effects", "No effect (pure noise)", "2019 survey (real data)"],
           loc="lower center", ncol=4, fontsize=7.5, handlelength=2.5)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig2_stability.png", dpi=300); fig.savefig(OUT / "fig2_stability.pdf")
plt.close(fig)

# ---- Figure 3: recovery of the true top driver -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, sc, title in zip(axes, ("linear", "nonlinear"), ("Linear effects", "Nonlinear effects")):
    for m, mk in (("LR", "o"), ("RF", "^"), ("XGB", "s")):
        d = r[r.scenario == sc].groupby("n")[f"top1_{m}"].mean()
        ax.plot(d.index, d.values, color=C[m], marker=mk, ms=5, label=NAME[m])
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=9, color=INK); ax.set_xlabel("Sample size (n)")
axes[0].set_ylabel("Share of samples where SHAP\nranks the true top driver first")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig3_top_driver.png", dpi=300); fig.savefig(OUT / "fig3_top_driver.pdf")
plt.close(fig)
print("figures written to", OUT)
