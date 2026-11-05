"""
Audit of the 2019 survey data (Data.sav / Data2.sav).
Reads the original SPSS files read-only; writes nothing back to them.

Usage: python -I audit_2019.py SKAFI_DIR OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
d1, m1 = pyreadstat.read_sav(src / "Data.sav")
d2, m2 = pyreadstat.read_sav(src / "Data2.sav")
R = {}

# ---------------------------------------------------------------- 1. files
shared = [c for c in d1.columns if c in d2.columns and c != "sector"]
same = all(((d1[c] == d2[c]) | (d1[c].isna() & d2[c].isna())).all() for c in shared)
R["files"] = {
    "Data.sav": list(d1.shape), "Data2.sav": list(d2.shape),
    "shared_columns_identical_row_by_row": bool(same),
    "only_in_Data": sorted(set(d1.columns) - set(d2.columns)),
    "only_in_Data2": sorted(set(d2.columns) - set(d1.columns)),
}

# ---------------------------------------------------------------- 2. target
R["target"] = {
    "Adpt_counts": d2["Adpt"].value_counts().to_dict(),
    "Adpt_value_labels": m2.variable_value_labels.get("Adpt"),
    "Adpt_equals_CCusage_recoded": bool(((d2["CCusage"] == 1).astype(int) == d2["Adpt"]).all()),
}

# ---------------------------------------------------------------- 3. sample profile
def tab(col, meta=m1, df=d1):
    lab = meta.variable_value_labels.get(col, {})
    vc = df[col].value_counts().sort_index()
    return {str(lab.get(k, k)): int(v) for k, v in vc.items()}

R["profile"] = {c: tab(c) for c in ["Job", "ITdecisionsmaking", "Numofemp", "revenue", "sector"]}
R["profile"]["adoption_by_size"] = (
    d1.assign(Adpt=d2["Adpt"]).groupby("Numofemp")["Adpt"].agg(["count", "mean"]).round(3)
    .rename(index=m1.variable_value_labels["Numofemp"]).to_dict(orient="index"))

# ---------------------------------------------------------------- 4. firms
name = d1["Companyname"].str.strip().str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
d1["_firm"] = name
per_firm = d1.groupby("_firm").agg(n=("_firm", "size"),
                                   adopt_values=("CCusage", lambda s: sorted(s.unique().tolist())),
                                   size_values=("Numofemp", lambda s: sorted(s.unique().tolist())))
multi = per_firm[per_firm.n > 1]
R["firms"] = {
    "distinct_company_names_raw": int(d1["Companyname"].nunique()),
    "distinct_after_normalising": int(per_firm.shape[0]),
    "respondents_per_firm_distribution": per_firm.n.value_counts().sort_index().to_dict(),
    "firms_with_2plus_respondents": int(len(multi)),
    "respondents_in_those_firms": int(multi.n.sum()),
    "firms_where_respondents_disagree_on_adoption": int((multi.adopt_values.str.len() > 1).sum()),
    "firms_where_respondents_disagree_on_size": int((multi.size_values.str.len() > 1).sum()),
    "largest_clusters": per_firm.sort_values("n", ascending=False).head(8)["n"].to_dict(),
}
per_firm.to_csv(out / "firm_clusters.csv")

# ---------------------------------------------------------------- 5. structural missingness
adopt = d2["Adpt"] == 1
blocks = {
    "intention (Q15)": ["intendtousecclin12", "woulduseccin12"],
    "barriers incl. political/infra/government (Q16)": [f"barriers{i}" for i in range(1, 21)],
    "service type (Q17)": ["cctype1", "cctype2", "cctype3"],
    "security & privacy (Q19)": [f"securityandprivacy{i}" for i in range(1, 7)],
    "supplier computing support (Q20)": ["SCS1", "SCS2", "SCS3", "SCS4"],
}
R["asked_of"] = {}
for b, cols in blocks.items():
    answered = d1[cols].notna().any(axis=1)
    R["asked_of"][b] = {"adopters_answering": int((answered & adopt).sum()),
                        "non_adopters_answering": int((answered & ~adopt).sum())}

# ---------------------------------------------------------------- 6. reliability
def alpha(X):
    X = X.dropna()
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))), len(X)

scales = {
    "Relative advantage (ccbenefit1-12)": [f"ccbenefit{i}" for i in range(1, 13)],
    "Cost (costeffect1-11, raw)": [f"costeffect{i}" for i in range(1, 12)],
    "Cost (1-8 + 11, cost-saving items only)": [f"costeffect{i}" for i in (1, 2, 3, 4, 5, 6, 7, 8, 11)],
    "Compatibility (1-7, raw)": [f"compatibility{i}" for i in range(1, 8)],
    "Compatibility (1-4,7; negatively worded 5-6 dropped)": [f"compatibility{i}" for i in (1, 2, 3, 4, 7)],
    "Complexity (1-5, raw)": [f"comlexity{i}" for i in range(1, 6)],
    "Complexity (1,2,4,5; positively worded 3 dropped)": [f"comlexity{i}" for i in (1, 2, 4, 5)],
    "Trialability (1-4)": [f"trialability{i}" for i in range(1, 5)],
    "Top management support (1-4)": [f"TopManSup{i}" for i in range(1, 5)],
    "Innovativeness (creativity1-3)": ["creativity1", "creativity2", "creativity3"],
    "Technological experience (competandtechexp1-9)": [f"competandtechexp{i}" for i in range(1, 10)],
    "Competitive pressure (LOC1-3)": ["LOC1", "LOC2", "LOC3"],
    "Security & privacy (1-6, adopters only)": [f"securityandprivacy{i}" for i in range(1, 7)],
    "Supplier support (SCS1-4, adopters only)": ["SCS1", "SCS2", "SCS3", "SCS4"],
    "Context barriers (barriers18-20, non-adopters only)": ["barriers18", "barriers19", "barriers20"],
}
R["cronbach_alpha"] = {k: {"alpha": round(a, 3), "n": n} for k, (a, n) in
                       ((k, alpha(d1[v])) for k, v in scales.items())}

# item-total check for negatively worded items
neg = {}
for scale, items, test in [("compatibility", [f"compatibility{i}" for i in range(1, 8)], ["compatibility5", "compatibility6"]),
                           ("complexity", [f"comlexity{i}" for i in range(1, 6)], ["comlexity3"]),
                           ("cost", [f"costeffect{i}" for i in range(1, 12)], ["costeffect9", "costeffect10"])]:
    for t in test:
        rest = d1[[c for c in items if c != t]].mean(axis=1)
        neg[t] = round(float(d1[t].corr(rest)), 3)
R["negatively_worded_item_rest_correlation"] = neg

# ---------------------------------------------------------------- 7. reproduce SPSS logistic regression
P8 = ["CCbenefitSc", "CosteffectSC", "CmptbltSC", "ComplxtSC", "TrialabilitySC", "InnovtSC", "TchnExpSC", "CmptPrsrSC"]
fit = sm.Logit(d2["Adpt"], sm.add_constant(d2[P8])).fit(disp=0)
n = len(d2)
cs = 1 - np.exp(2 * (fit.llnull - fit.llf) / n)
R["logit_reproduction"] = {
    "LR_chi2": round(float(2 * (fit.llf - fit.llnull)), 3), "df": 8, "p": round(float(fit.llr_pvalue), 3),
    "minus2LL": round(float(-2 * fit.llf), 3), "cox_snell": round(float(cs), 3),
    "nagelkerke": round(float(cs / (1 - np.exp(2 * fit.llnull / n))), 3),
    "pct_correct_at_0.5": round(float(((fit.predict() >= .5) == d2["Adpt"]).mean() * 100), 1),
    "coef": fit.params.round(3).to_dict(), "p_values": fit.pvalues.round(3).to_dict(),
}

# Same model, adding the two other complete-case factor scores and the SME / decision-maker filters
def logit(df, cols):
    f = sm.Logit(df["Adpt"], sm.add_constant(df[cols])).fit(disp=0)
    nn = len(df); c = 1 - np.exp(2 * (f.llnull - f.llf) / nn)
    return {"n": nn, "adopters": int(df.Adpt.sum()), "p": round(float(f.llr_pvalue), 3),
            "nagelkerke": round(float(c / (1 - np.exp(2 * f.llnull / nn))), 3),
            "sig_at_.05": [k for k, v in f.pvalues.items() if k != "const" and v < .05]}

sme = d1["Numofemp"].isin([1, 2])                 # < 200 employees (closest available cut-off)
dm = d1["ITdecisionsmaking"] == 1
R["subsets"] = {
    "all_139_8predictors": logit(d2, P8),
    "all_139_plus_TopMgt": logit(d2, P8 + ["TopMgtSC"]),
    "SME_<200_employees": logit(d2[sme], P8),
    "IT_decision_makers": logit(d2[dm], P8),
    "SME_and_decision_makers": logit(d2[sme & dm], P8),
}

(out / "audit_results.json").write_text(json.dumps(R, indent=2, default=str))

# Clean analysis file for the ML pipeline (no names; firm ID kept for grouped CV)
firm_id = d1["_firm"].astype("category").cat.codes
clean = d2[P8 + ["TopMgtSC", "Adpt"]].copy()
clean["firm_id"] = firm_id
clean["sme_lt200"] = sme.astype(int)
clean["it_decision_maker"] = dm.astype(int)
clean["data_source"] = "REAL_2019_SURVEY"
clean.to_csv(out / "survey2019_analysis.csv", index=False)
print(json.dumps(R, indent=2, default=str))
