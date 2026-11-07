"""Summarise the correlation experiment (Table 6) from results/corr_raw.csv and the
moderate level taken from results/sim_raw_interventional.csv."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

kw = dict(keep_default_na=False, na_values=[""])
c = pd.read_csv(RESULTS / "corr_raw.csv", **kw)
m = pd.read_csv(RESULTS / "sim_raw_interventional.csv", **kw)
m = m[m.n.isin([139, 300, 1000])].copy(); m["corr"] = "moderate"
d = pd.concat([c, m]); d["gain_XGB"] = d.auc_XGB - d.auc_LR; d["gain_RF"] = d.auc_RF - d.auc_LR
cols = [f"{k}_{mm}" for k in ("auc", "stab", "dominance", "tau_truth", "top1", "top3") for mm in ("LR", "RF", "XGB")] + ["gain_RF", "gain_XGB"]
g = d.groupby(["scenario", "n", "corr"])[cols].mean().round(3)
g["reps"] = d.groupby(["scenario", "n", "corr"]).size()
g = g.reset_index()
g["o"] = g["corr"].map({"low": 0, "moderate": 1, "high": 2})
g = g.sort_values(["scenario", "n", "o"]).drop(columns="o")
g.to_csv(RESULTS / "corr_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(g[["scenario", "n", "corr", "reps", "stab_LR", "stab_RF", "stab_XGB", "tau_truth_LR", "tau_truth_RF",
         "tau_truth_XGB", "top1_LR", "top1_RF", "top1_XGB", "gain_XGB"]].to_string(index=False))
