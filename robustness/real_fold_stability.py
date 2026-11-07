"""Fold stability of SHAP rankings on the 2019 survey with the simulation protocol
(20 repetitions of stratified 5-fold CV, fixed hyperparameters; Section 5.1 and Fig. 2).

Usage:  python real_fold_stability.py PATH/TO/survey2019_analysis.csv [OUT_JSON]
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["PL_DATA"] = os.path.abspath(sys.argv[1])
from common import RESULTS
import permutation_lib as pl

OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "real_fold_stability_interventional.json")
st, auc = pl.fold_stab(pl.y, 20, 0, return_all=True)
res = {"fold_stability": {m: round(float(np.mean(v)), 3) for m, v in st.items()},
       "fold_stability_sd": {m: round(float(np.std(v, ddof=1)), 3) for m, v in st.items()},
       "simulation_protocol_auc": {m: round(float(np.mean(v)), 3) for m, v in auc.items()}}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
