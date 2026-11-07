"""Bootstrap stability of SHAP rankings on the 2019 survey (Table 3, observed values in Table 4).

Requires the CONFIDENTIAL analysis file produced by real_data/audit_2019.py
(survey2019_analysis.csv). That file is not in this repository and must never be committed.
Only the aggregate output (JSON) is written.

Usage:  python real_bootstrap.py PATH/TO/survey2019_analysis.csv [OUT_JSON] [--path-dependent]
--path-dependent keeps the default path-dependent TreeSHAP (sensitivity check).
"""
import json, os, sys, warnings
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_ra

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PD = "--path-dependent" in sys.argv
DATA = args[0]
OUT = args[1] if len(args) > 1 else str(RESULTS / ("real_bootstrap_pathdependent.json" if PD else "real_bootstrap_interventional.json"))
ra = load_ra(interventional=not PD)
df = pd.read_csv(DATA); X = df[ra.FEATS].to_numpy(float); y = df["Adpt"].to_numpy(int)
stab, ranks = ra.bootstrap_stability(X, y)
for name, R in ranks.items():
    R = np.vstack(R)
    # mean Kendall tau over all pairs of the 200 refits (the statistic reported in the paper)
    stab[name]["tau_all200"] = float(np.mean([stats.kendalltau(a, b)[0] for a, b in combinations(R, 2)]))
json.dump(stab, open(OUT, "w"), indent=1)
print(json.dumps({k: v.get("tau_all200") for k, v in stab.items()}, indent=1))
