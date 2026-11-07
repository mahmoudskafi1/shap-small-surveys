"""Check with 10,000 simulated respondents (Section 4.2): can each model recover the true
importance when sampling error is small? 2 scenarios x 10 replications. Synthetic data only.

Usage:  python large_n_check.py [N_JOBS] [OUT_JSON]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RESULTS


def run(sc, rep):
    sys.path.insert(0, HERE)
    from common import load_sim
    sim = load_sim(); sim.NL_SCALE = sim._scale_for_nonlinear(); truth = sim.true_importance(sc)
    X, y = sim.make_data(10_000, sc, 900 + rep); Xte, _ = sim.make_data(1_000, sc, 950 + rep)
    out = []
    for name, m in sim.models(rep).items():
        m.fit(X, y); imp = sim.shap_imp(name, m, X[:1000], Xte)
        out.append(dict(sc=sc, rep=rep, model=name, top1=int(np.argmax(imp) == np.argmax(truth)),
                        top3=len(set(np.argsort(-imp)[:3]) & set(np.argsort(-truth)[:3])) / 3,
                        tau=float(kendalltau(imp, truth)[0])))
    return out


if __name__ == "__main__":
    NJ = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    OUT = sys.argv[2] if len(sys.argv) > 2 else str(RESULTS / "large_n_check.json")
    res = Parallel(n_jobs=NJ)(delayed(run)(s, r) for s in ("linear", "nonlinear") for r in range(10))
    rows = [x for r in res for x in r]; json.dump(rows, open(OUT, "w"))
    d = pd.DataFrame(rows)
    print(d.groupby(["sc", "model"])[["top1", "top3", "tau"]].agg(["mean", "min", "max"]).round(2).to_string())
