# Synthetic RAPI Demonstration Dataset

This folder contains a reproducible synthetic dataset for the two-notebook
Regime-Aware Process Intelligence workflow.

The data are anonymized and intentionally use generic feature names. They are
designed to resemble a scaled, unstable bioprocess without exposing any real
facility, equipment, organism, vendor, or production data.

## Files

- `time_series/RUN_###.csv`: raw per-run process time series sampled every
  15 minutes over a two-week production window.
- `supplemental_time_series/RUN_###_GR##_supplemental.csv`: supplemental
  run-aligned time-series metrics used to create `feature_9` and
  `feature_10_ext_metric` features.
- `run_metadata.csv`: run-level metadata and the target response consumed by
  the rolling notebook.
- `run_treatment_metrics.csv`: optional run-level treatment metrics consumed by
  both notebooks as `feature_16` and `feature_17`.
- `external_run_metrics.csv`: optional external run-level metrics consumed by
  the rolling notebook as `feature_18` and `feature_19`.
- `synthetic_ground_truth.csv`: generator-only documentation of latent regimes
  and target contributions. The notebooks do not need this file.

## Intended Demonstration

The target, `target_metric_primary`, is a conversion-like rate bounded between
0.0 and 0.075. It is generated from three categories of structure:

- Persistent cross-regime drivers: `feature_12`, `feature_13`, `feature_16`,
  `feature_17`, and `feature_18`.
- Regime-dependent drivers: early runs emphasize `feature_1_block` and
  `feature_3`; middle runs emphasize a nonlinear `feature_8` response and
  its interaction with `feature_9`; late runs emphasize `feature_4_1`,
  `feature_6`, and treatment delta.
- Intentionally weak or irrelevant signals: `feature_15`, `feature_19`, and
  several trace-level oscillations/noise components.

This lets the rolling notebook surface changing local driver importance while
the global notebook can still recover persistent structure across all regimes.

## Regeneration

From the repository root:

```bash
python generate_synthetic_rapi_data.py
```

The generator overwrites only CSV files inside `pipeline_inputs/time_series`
and `pipeline_inputs/supplemental_time_series`, then rewrites the run-level
input CSVs listed above.
