# Robustness analyses added in the revision

This folder contains the analyses added during the revision of the paper. The original
scripts in `real_data/`, `simulation/` and `figures/` are unchanged. The scripts here import
them and change only what is described below. Package versions are the ones in
`requirements.txt`. Run every command from inside this folder.

**Confidentiality.** The 2019 survey data (`*.sav`, `survey2019_analysis.csv`) are not in
this repository and must not be added. The three real-data scripts take the path to the
locally held analysis file (produced by `real_data/audit_2019.py`) as an argument and write
aggregate statistics only.

## 1. Interventional SHAP for all models

`interventional_shap.py` replaces the SHAP functions of the original scripts, so that random
forest and XGBoost are explained with interventional TreeSHAP (background of up to 100
training cases), in the same way as logistic regression and the true importance.

| Script | Paper | Output in `results/` |
|---|---|---|
| `run_simulation.py` | Table 5, Figs 1–3 | `sim_raw_interventional.csv` (synthetic) |
| `large_n_check.py` | Section 4.2, check with 10,000 respondents | `large_n_check.json` (synthetic) |
| `real_bootstrap.py DATA` | Table 3, Table 4 (observed) | `real_bootstrap_interventional.json` |
| `real_bootstrap.py DATA --path-dependent` | Limitations (sensitivity check) | `real_bootstrap_pathdependent.json` |
| `real_fold_stability.py DATA` | Section 5.1, Fig. 2 | `real_fold_stability_interventional.json` |
| `run_permutation_null.py DATA` | Table 4 (null) | `perm_null_interventional.jsonl` |
| `summarise_permutation.py` | Table 4 | printed |
| `make_figures.py` | Figs 1–3 | `figures/` |

The bootstrap agreement reported in the paper is `tau_all200`, the mean Kendall τ over all
pairs of the 200 refits.

## 2. Organisation-aware analyses (Section 5.1)

`grouped_cv.py DATA` keeps the respondents of each organisation together: stratified grouped
folds in the outer and inner loops of the nested cross-validation, in the permutation test and
in the fold-stability computation, and a bootstrap that resamples organisations. Respondents
with a blank employer field are treated as separate organisations. Output:
`results/real_grouped_cv.json` (aggregate statistics only).

`grouped_null.py DATA` computes the organisation-aware noise baseline: for each of the same 100
permutations of the outcome used for Table 4, the 200-refit bootstrap that resamples
organisations is repeated. Output: `results/grouped_null.jsonl` (aggregate statistics only).
`python grouped_null.py --summary` prints the comparison with the observed organisation-level
bootstrap (Section 5.1).

`restricted_permutation.py` repeats both permutation tests with a restricted (block)
permutation that keeps the respondents of each organisation together (Winkler et al., 2015):
the outcomes of whole organisations are exchanged between organisations of the same size, so
single respondents are shuffled freely, the pairs from the same organisation are exchanged as
pairs, and the two organisations of unique size (4 and 18 respondents) keep their outcomes.
These are the organisation-aware results reported in Section 5.1.

| Command | Output in `results/` |
|---|---|
| `python restricted_permutation.py auc DATA` | `restricted_auc_test.json` (logistic regression AUC, 500 permutations) |
| `python restricted_permutation.py boot DATA` | `restricted_null.jsonl` (noise baseline for the organisation-level bootstrap, 100 permutations) |
| `python restricted_permutation.py --summary` | printed comparison with the observed values |

The results of `grouped_null.py`, which shuffles the outcome across respondents, are kept for
comparison.

## 3. Correlation between constructs (Table 6)

| Script | Purpose |
|---|---|
| `correlation_lib.py` | Low (independent), moderate (2019 matrix) and high (2019 matrix averaged with 0.6) levels; the signal is rescaled at each level so that the best achievable AUC stays at 0.778 |
| `run_correlation.py` | Runs the low and high levels (n = 139, 300, 1,000; 100 replications); the moderate level is the main simulation |
| `summarise_correlation.py` | Writes `results/corr_summary.csv` (Table 6) |

## 4. Tuning check for the simulation (Section 5.2, Limitations)

`run_tuning.py` reruns the nonlinear scenario at n = 139, 300 and 1,000 on the same 100
synthetic datasets per size as the main simulation (same seeds), but selects the
hyperparameters of every model by a stratified 5-fold grid search on ROC-AUC inside each
training fold. The grids contain the fixed values of the main simulation:

| Model | Grid |
|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10} |
| Random forest (250 trees) | maximum depth in {2, 4, unlimited} x minimum leaf size in {3, 5, 10} |
| XGBoost (200 trees, learning rate 0.05) | maximum depth in {1, 2, 3} x L2 penalty in {1, 5, 10} |

| Command | Output in `results/` |
|---|---|
| `python run_tuning.py results/sim_tuned_nonlinear.csv 6600` | `sim_tuned_nonlinear.csv` (synthetic; one row per dataset, with the hyperparameters selected in every fold) |
| `python run_tuning.py --summary results/sim_tuned_nonlinear.csv` | `sim_tuned_nonlinear_summary.csv` (tuned and fixed results on the same datasets) and the printed change in the ML gain |

## 5. Checks of the block permutation (Section 4.1, Section 5.1, Limitations)

In `restricted_permutation.py`, the organisations with 4 and 18 respondents have no partner of
the same size, so their outcomes stay fixed in every permutation. `org_permutation_checks.py`
checks what this choice does. Like the other real-data scripts, it needs the confidential
analysis file, which is not in this repository, and it writes aggregate statistics only.

| Command | Output in `results/` |
|---|---|
| `python org_permutation_checks.py sub DATA` | `org_sub117.json`: AUC permutation test and organisation-level bootstrap without the two organisations (n = 117), where every organisation can exchange its outcomes |
| `python org_permutation_checks.py subnull DATA results/org_sub117_null.jsonl 100 6000` | `org_sub117_null.jsonl`: noise baseline for that bootstrap (100 permutations x 200 refits; resumable) |
| `python org_permutation_checks.py calib DATA results/org_calibration.jsonl 300 6000` | `org_calibration.jsonl`: Type I error of the AUC permutation test on 900 datasets with a simulated outcome that is unrelated to the constructs but correlated within organisations (intra-class correlation 0, 0.2, 0.5); each row holds the p values of the block permutation and of the shuffle across respondents |
| `python org_permutation_checks.py --summary results` | printed summary, saved as `org_permutation_checks_summary.txt` |

## Running times

The simulation scripts take several hours on two cores. They can be resumed: pass a time
budget in seconds as the optional argument, and run the same command again to continue.
For example:

```
python run_correlation.py low,high results/corr_raw.csv 6600
```
