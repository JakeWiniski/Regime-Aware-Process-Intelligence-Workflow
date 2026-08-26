# RAPI: Regime-Aware Process Intelligence - Learning Under Drift in Non-Stationary Process Systems
[![DOI](https://zenodo.org/badge/1244655605.svg)](https://doi.org/10.5281/zenodo.20358130)

This repository demonstrates a **Regime-Aware Process Intelligence (RAPI)** workflow for analyzing complex production systems whose behavior is nonstationary, only partially observed, and still evolving while operating at scale.

The workflow is motivated by a real-world class of problem: a scaled biological or cyber-physical production process where the output response is observed, many process features are measured, and the system is drifting before a stable validated baseline exists. In that setting, conventional process-control approaches that assume stationarity can be misleading. The more useful question is often not "is the process out of control relative to a known baseline?" but:

> What is changing locally, and what remains consistently important despite that change?

This repository answers that question with two complementary notebooks:

1. `Rolling_Surrogate_Regime_Analysis.ipynb`
2. `Global_Model.ipynb`

The first notebook detects and characterizes local regime shifts. The second notebook identifies stable global structure and cross-regime drivers.

---

## Use Case And Rationale

The workflow is designed for systems with several difficult properties:

- The target response is observed, but not all true drivers are measured.
- The observed feature space is high-dimensional and mixed: continuous variables, categorical variables, time-series sensor traces, engineered descriptors, treatment metrics, and external context.
- The process is nonstationary: feature distributions, target behavior, and feature-target relationships can drift over time.
- Some drivers may matter only within particular regimes, while others may remain important across the full operating history.
- The process may be too immature for classical statistical process control assumptions to hold.

The analytical strategy is therefore split into two views:

| Question | Notebook | Methodological Role |
|---|---|---|
| What is changing locally? | Rolling surrogate analysis | Fit local models over rolling windows and monitor residuals, fit, and drivers |
| What remains stably important? | Global model | Use dimensionality reduction and repeated-fold stability selection to identify persistent structure |

Together, the notebooks support practical reasoning in unstable systems: they distinguish **local regime-dependent behavior** from **global cross-regime drivers**.

## Synthetic Demonstration Dataset

The synthetic dataset contains:

- 120 production runs.
- Six anonymized production units.
- Raw per-run time-series files sampled every 15 minutes.
- Supplemental time-series metrics.
- Run-level metadata.
- Continuous and categorical variables.
- Linear, nonlinear, interaction, and regime-dependent relationships.
- Persistent cross-regime drivers.
- Regime-specific drivers that drift in importance over time.
- Intentionally weak or unimportant features.

The target, `target_metric_primary`, is a conversion-like response bounded between 0.0 and 0.075.

The file `pipeline_inputs/synthetic_ground_truth.csv` documents the hidden synthetic regimes and target-contribution components. The notebooks do not require this file; it exists to make the demonstration auditable.

To regenerate the synthetic inputs:

```bash
python generate_synthetic_rapi_data.py
```

---

## Running The Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:

1. `Rolling_Surrogate_Regime_Analysis.ipynb`
2. `Global_Model.ipynb`

The first notebook creates the feature table consumed by the second:

```text
pipeline_outputs/features/feature_table.csv
```

If you regenerate the synthetic dataset, rerun the rolling notebook before rerunning the global notebook.

---

## Notebook 1: Rolling Surrogate Regime Analysis

The rolling notebook treats each production run as both a production event and an observation in an evolving experiment.

### 1. Raw Time-Series Alignment

Each raw run file is loaded, timestamped, and cropped to the run window defined in `pipeline_inputs/run_metadata.csv`. Signals are smoothed and standardized into a consistent per-run structure.

The expected raw inputs include:

- `pipeline_inputs/time_series/*.csv`
- `pipeline_inputs/supplemental_time_series/*.csv`
- `pipeline_inputs/run_metadata.csv`
- optional run-level treatment and external metric files

### 2. Time-Window Feature Construction

Each run is segmented into three configurable time windows:

- `WINDOW_1`
- `WINDOW_2`
- `WINDOW_3`

Within each window, each signal is summarized using compact descriptors such as:

- median level,
- slope,
- residual variability,
- lag-1 autocorrelation,
- dominant frequency.

This preserves temporal context while converting raw traces into a wide run-level feature table.

### 3. Coupling Features

The notebook also computes lag-aware coupling features between selected signals. These are intended to capture changes in system responsiveness or coordination that may not be visible from a single signal alone.

### 4. Run-Level Context Integration

Time-series features are merged with run-level metadata, categorical variables, treatment-like metrics, target values, and external context.

The resulting table has one row per run and is saved as:

```text
pipeline_outputs/features/feature_table.csv
```

### 5. Rolling Local Surrogate Modeling

The final section trains a sequence of local Random Forest surrogate models. Each model is fit on a rolling window of recent runs and predicts the next run out of sample.

The notebook tracks:

- out-of-sample residuals,
- inner cross-validation RMSE,
- out-of-bag R²,
- local feature driver snapshots.

These outputs are diagnostic. The model is not treated as the final predictor; it is treated as a measurement device for local structure.

### 6. Interpreting Rolling Outputs

Signals of possible regime change include:

- residuals becoming biased or unstable,
- OOB R² rising or falling sharply,
- changes in which features appear locally important,
- changes in signal coupling or responsiveness.

The key point is that model instability is not merely a problem to hide. In this workflow, instability is often the signal.

---

## Notebook 2: Global Model

The global notebook starts from the rolling feature table and asks a different question:

> Which features remain consistently important across the operating history, despite local regime drift?

### 1. Feature Integration

The notebook loads:

```text
pipeline_outputs/features/feature_table.csv
```

It cleans missing values, converts numeric-like fields, joins optional manual metrics if present, and prepares a global run-level analysis table.

### 2. PCA Structural Mapping

The notebook applies PCA to selected process-feature groups. PCA is used as a structural map of high-dimensional process variation, not as a causal explanation.

The PCA section exports:

- PCA scores,
- feature-PC correlations,
- cosine-squared contribution tables,
- top feature contributors by PC.

The modeling dataset can include PCs in either of two ways:

1. manual PC selection, or
2. automatic selection by cumulative explained-variance threshold.

For example, an 80% threshold includes the minimum number of PCs required to explain at least 80% of the selected process-feature variance.

### 3. Targeted Feature Engineering

The notebook includes a small number of explicit engineered features. These are intentionally limited and hypothesis-driven, such as:

- nonlinear interactions,
- treatment deltas,
- standardized interaction terms.

This keeps the model interpretable while allowing specific domain hypotheses to be represented.

### 4. Stability-Selected Union Feature Selection

This is the core global-model improvement.

The feature-selection process combines two complementary methods:

- **Elastic Net**, which captures stable linear or monotonic structure.
- **Random Forest feature importance**, which captures nonlinear and interaction-driven structure.

For each resampled fold, the notebook computes:

```text
Elastic Net selected features ∪ Random Forest selected features
```

Then it repeats this process across multiple folds and repeats. A feature is retained only if it is selected by either method in at least a configurable fraction of folds.

Default controls:

```python
STABILITY_N_SPLITS = 5
STABILITY_N_REPEATS = 5
STABILITY_MIN_SELECTION_RATE = 0.60
```

The main interpretive output is:

```python
stable_feature_summary
```

which includes:

- `enet_selection_rate`
- `rf_selection_rate`
- `union_selection_rate`
- `mean_rf_importance`
- `std_rf_importance`
- `stable_selected`

This table is designed to answer the global question directly: what is repeatedly important across resampled views of the full operating history?

### 5. Fold-Local Validation

The validation block refits the stability-selection process inside each training fold before fitting the Random Forest model. This avoids leakage and provides a more honest estimate of workflow performance.

Performance is reported, but it is secondary to interpretability. The central artifact remains the stable feature set.

### 6. Final Regression Model And SHAP

The final regression Random Forest is trained on the stability-selected feature matrix:

```python
X_selected
```

SHAP is then used to explain the fitted model. SHAP values should be interpreted as model-based evidence, not causal proof. The strongest candidates for practical investigation are features that are both:

- frequently selected in `stable_feature_summary`, and
- influential in SHAP summaries.

### 7. Contextual Anomaly Scoring

The notebook also provides diagnostic anomaly scoring:

- PCA-space anomaly: unusual operating state.
- residual anomaly: model disagreement with observed target.

These scores help prioritize runs for human review. They are not automated failure labels.

### 8. Optional Classification View

The notebook includes an optional binary classification branch that reframes the continuous target as low vs high outcome state.

This branch uses the same stability-selection philosophy:

```text
Logistic Elastic Net selected features ∪ Random Forest classifier selected features
```

with repeated-fold stability filtering. The classification outputs provide a complementary view of which features distinguish outcome states.

---

## Key Outputs

Rolling notebook outputs:

```text
pipeline_outputs/features/feature_table.csv
pipeline_outputs/features/feature_table_full.csv
pipeline_outputs/features/training_features.csv
pipeline_outputs/features/training_target.csv
pipeline_outputs/features/model_driver_snapshots.csv
```

Global notebook outputs:

```text
pipeline_outputs/global_model_outputs/pca_scores.csv
pipeline_outputs/global_model_outputs/feature_vs_pc_cos2.csv
pipeline_outputs/global_model_outputs/top_feature_contributors_by_pc_cos2.csv
pipeline_outputs/global_model_outputs/pearson_feature_vs_pc_r_and_p.csv
pipeline_outputs/global_model_outputs/training_data_prepared.csv
pipeline_outputs/global_model_outputs/training_data_engineered.csv
```

Some model explanation outputs, such as SHAP values and anomaly reports, are generated when the corresponding notebook cells are run.

---

## How To Interpret The Two-Notebook Workflow

Read the notebooks together:

- The rolling notebook identifies local drift, local driver changes, and unstable operating periods.
- The global notebook identifies stable cross-regime drivers and structural process axes.

Useful interpretation patterns:

- A feature that appears in rolling driver shifts but not global stability selection may be regime-specific.
- A feature that appears in global stability selection but not rolling driver snapshots may be a persistent background constraint.
- A feature that appears in both deserves special practical attention.
- A run that is unusual in PCA space and has a large residual deserves human review.

This workflow is designed for human-in-the-loop process intelligence. It does not claim causal certainty or autonomous control. It creates structured evidence that helps operators, engineers, and scientists reason about evolving systems before full stabilization is achieved.

---

## Design Principles

- **Regime awareness over stationarity:** assume relationships can change until proven otherwise.
- **Stability over one-off importance:** global importance must persist across resampled folds.
- **Explainability over optimization:** models are tools for understanding process structure.
- **Change as signal:** drift, instability, and model failure can be primary observables.
- **Multi-source integration:** time series, metadata, categorical context, and external metrics are analyzed together.
- **Human-in-the-loop interpretation:** outputs support investigation and decision-making, not automatic control.

---

## Notes

The synthetic dataset is not intended to represent any real facility or proprietary process. It is designed to demonstrate the workflow mechanics and analytical logic in a reproducible, anonymized form.

The methods here are intentionally practical. They are meant to help reason about messy production systems where perfect causal identification is not available, but where structured evidence can still improve situational awareness and learning velocity.

## Disclaimer

This repository is intended solely as an educational and methodological demonstration of a regime-aware analytical workflow. All datasets included in this project are synthetic and were generated specifically for illustrative purposes. The data do not originate from any real manufacturing process, biological system, company operation, customer program, or experimental dataset.

The workflows, analyses, and examples presented here are designed to demonstrate general principles of statistical learning, process interpretation, and adaptive analytical strategies under nonstationary conditions. Any resemblance between the synthetic examples and real-world systems is coincidental.

This repository does not disclose proprietary processes, confidential information, trade secrets, internal methodologies, or unpublished data from any company, organization, employer, collaborator, or client. The concepts presented represent generalized educational examples and publicly shareable methodological approaches only.

## Intended Use

This project is shared as an open methodological example and portfolio artifact. It is intended to support learning, discussion, adaptation, and extension by others interested in process development, manufacturing analytics, experimental learning, and nonstationary systems.

## License

This project is released under the Apache License 2.0.

You are free to use, modify, and distribute this work in accordance with the license terms. Attribution is appreciated. Please cite the repository and associated DOI when using this work in research, publications, or derivative projects.

## Citation

If you use this workflow, methodology, code, or derivative work in research, publications, presentations, or other projects, please cite this repository.

**APA**

Winiski, J. (2026). *RAPI: Regime-Aware Process Intelligence* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20358131

## AI Use Disclosure

This project was developed with assistance from generative AI and coding-agent tools. AI was used for coding, debugging, and development support; analytical design, validation, interpretation, and final decisions were performed by the author.


