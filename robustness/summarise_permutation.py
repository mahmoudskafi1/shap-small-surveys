"""Table 4: observed statistics against the permutation null.
p = (number of null values >= observed + 1) / (K + 1). For the first-place and top-three
shares, the observed value is the share of top management support and the null value is
the largest share of any construct in that permutation."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

P = [json.loads(L) for L in open(RESULTS / "perm_null_interventional.jsonl")]
B = json.load(open(RESULTS / "real_bootstrap_interventional.json"))
F = json.load(open(RESULTS / "real_fold_stability_interventional.json"))["fold_stability"]
TM = "Top management support"
rows = []
for stat in ("first", "top3", "boot_tau", "fold_tau"):
    for m in ("LR", "RF", "XGB"):
        if stat == "first":
            obs = B[m]["share_ranked_first"][TM]; null = [p["boot"][m]["max_first"] for p in P]
        elif stat == "top3":
            obs = B[m]["share_in_top3"][TM]; null = [p["boot"][m]["max_top3"] for p in P]
        elif stat == "boot_tau":
            obs = B[m]["tau_all200"]; null = [p["boot"][m]["tau"] for p in P]
        else:
            obs = F[m]; null = [p["fold"][m] for p in P]
        null = np.array(null)
        pval = (np.sum(null >= obs) + 1) / (len(null) + 1)
        rows.append((stat, m, obs, null.mean(), np.percentile(null, 95), null.max(), pval))
print(f"{'statistic':10s} {'model':5s} {'obs':>6s} {'mean':>6s} {'p95':>6s} {'max':>6s} {'p':>5s}")
for r in rows:
    print(f"{r[0]:10s} {r[1]:5s} {r[2]:6.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:5.2f}")
for m in ("LR", "RF", "XGB"):
    print(m, "TopMgt first under null:", round(np.mean([p["boot"][m]["tm_first"] for p in P]), 3),
          "top three:", round(np.mean([p["boot"][m]["tm_top3"] for p in P]), 3))
