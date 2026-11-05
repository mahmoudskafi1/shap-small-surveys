# When does explainable machine learning add value to small technology-adoption surveys?

Code, synthetic data and aggregate results for the paper

> Skafi, M. *When Does Explainable Machine Learning Add Value to Small Technology-Adoption Surveys? A Reanalysis and Calibrated Simulation.* Manuscript submitted to *Information Systems Frontiers*.

The paper compares logistic regression, random forest and XGBoost on a published survey of 139 respondents on cloud computing adoption in Lebanon (Skafi, Yunis & Zekri, 2020, *IEEE Access*, https://doi.org/10.1109/ACCESS.2020.2987331), measures the stability of SHAP importance rankings, and runs a simulation of 1,800 synthetic datasets calibrated to that survey.

## What is (and is not) included

| Included | Not included |
|---|---|
| All analysis and simulation code | The 2019 survey responses (respondent-level data) |
| The full synthetic simulation output (`simulation/sim_raw.csv`, 1,800 datasets) | The analysis file built from them (`survey2019_analysis.csv`) |
| Aggregate results of the real-data analysis | Any company names or respondent identifiers |
| The aggregate calibration values taken from the 2019 data | |
| Figures 1–3 (PDF and PNG) | |

The survey responses were collected under a promise of confidentiality and are available from the author on reasonable request. Everything in the simulation can be rerun from this repository alone.

## Repository structure

```
real_data/
  audit_2019.py            builds the analysis file from the confidential SPSS files (Data.sav, Data2.sav; not included)
  real_analysis.py         nested cross-validation, corrected resampled t-test, permutation test,
                           bootstrap SHAP stability, sensitivity analysis (Tables 2-3, Section 5.1)
  real_results.json        aggregate results of real_analysis.py
  nested_cv_summary.csv    Table 2 (mean and SD over 50 outer test folds)
  real_fold_stability.json fold-to-fold SHAP rank agreement on the 2019 data (diamond in Figure 2)
simulation/
  calibrated_sim.py        calibrated simulation (Table 4, Figures 1-3)
  calibration_2019.json    aggregate values from the 2019 data: construct correlations, item loadings,
                           item response distributions, reliabilities, coefficients, adoption rate
  sim_raw.csv              SYNTHETIC output: one row per simulated dataset (3 scenarios x 6 sizes x 100 reps)
  sim_summary.csv          aggregated simulation results
  sim_meta.json            oracle AUCs and true importances of the data-generating process
figures/
  make_figures_nocap.py    draws Figures 1-3
  fig1_auc_gain.*, fig2_stability.*, fig3_top_driver.*
```

## Requirements

Python 3.13. Install the packages with

```
pip install -r requirements.txt
```

The versions of scikit-learn, XGBoost, SHAP, NumPy and SciPy are the ones reported in the paper (Section 4.3).

## How to reproduce

**Simulation (no confidential data needed).** From the repository root:

```
python simulation/calibrated_sim.py simulation_out 100
```

The second argument is the number of replications per condition; the paper uses 100 (the script's default is 50). Seeds are fixed, so the output reproduces `simulation/sim_raw.csv`. A full run fits 3 models to 1,800 datasets with SHAP and may take several hours.

**Figures.**

```
python figures/make_figures_nocap.py
```

This reads `simulation/sim_raw.csv` and the files in `real_data/`, and writes Figures 1-3 to `figures/`. Pass a folder name as an argument to write them elsewhere.

**Real-data analysis (requires the confidential survey files).**

```
python real_data/audit_2019.py PATH_TO_SPSS_FILES audit_out
python real_data/real_analysis.py audit_out/survey2019_analysis.csv real_out
```

## Note on reading `sim_raw.csv`

One scenario is called `null`. pandas reads the text "null" as a missing value by default, so read the file with

```python
pd.read_csv("simulation/sim_raw.csv", keep_default_na=False, na_values=[""])
```

otherwise the 600 null-scenario rows appear to have no scenario label.

## Licence

Code: MIT licence (see `LICENSE`). Synthetic data and figures: free to reuse with citation.

## Citation

Please cite the paper and this archive (see `CITATION.cff`).
