"""Generate a synthetic dataset for the Regime-Aware Process Intelligence notebooks.

The generated files intentionally match the raw-input schema expected by
Rolling_Surrogate_Regime_Analysis.ipynb:

  pipeline_inputs/
    run_metadata.csv
    run_treatment_metrics.csv
    external_run_metrics.csv
    synthetic_ground_truth.csv
    time_series/RUN_###.csv
    supplemental_time_series/RUN_###_GR##_supplemental.csv

The data are anonymized, but the structure mirrors a non-stationary scaled
bioprocess: six units, overlapping two-week production runs, raw time-series
signals, supplemental time-series metrics, categorical run descriptors, and a
conversion-like target with both persistent and regime-dependent drivers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pipeline_inputs"
TS_DIR = INPUT_DIR / "time_series"
SUPP_DIR = INPUT_DIR / "supplemental_time_series"

RNG = np.random.default_rng(20260523)

N_RUNS = 120
N_UNITS = 6
SAMPLE_FREQ = "15min"


def seasonal_phase(i: int) -> float:
    """Annual-ish seasonality over the synthetic production campaign."""
    return 2.0 * math.pi * i / N_RUNS


def regime_label(i: int) -> str:
    """Piecewise regimes with soft drift inside each period."""
    if i < 38:
        return "regime_a"
    if i < 78:
        return "regime_b"
    return "regime_c"


def smooth_noise(n: int, scale: float, persistence: float = 0.92) -> np.ndarray:
    """Small AR(1)-like noise for process traces."""
    out = np.zeros(n)
    eps = RNG.normal(0.0, scale, n)
    for j in range(1, n):
        out[j] = persistence * out[j - 1] + eps[j]
    return out


def pulse_train(
    n: int,
    count: int,
    amplitude: float,
    width_min: int,
    width_max: int,
    signed: bool = False,
) -> np.ndarray:
    """Return sparse Gaussian pulses for transient process disturbances."""
    x = np.arange(n)
    out = np.zeros(n)
    for _ in range(count):
        center = RNG.integers(0, n)
        width = RNG.integers(width_min, width_max + 1)
        sign = RNG.choice([-1.0, 1.0]) if signed else 1.0
        height = sign * RNG.normal(amplitude, amplitude * 0.25)
        out += height * np.exp(-0.5 * ((x - center) / max(width, 1)) ** 2)
    return out


def low_frequency_wander(n: int, scale: float, knots: int = 7) -> np.ndarray:
    """Smooth, non-periodic drift created by interpolating random anchor points."""
    xp = np.linspace(0, n - 1, knots)
    yp = RNG.normal(0.0, scale, knots)
    return np.interp(np.arange(n), xp, yp)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def build_run_latents(i: int) -> dict[str, object]:
    """Create run-level latent factors and the target-generation components."""
    phase = seasonal_phase(i)
    season_sin = math.sin(phase)
    season_cos = math.cos(phase)
    regime = regime_label(i)
    drift = i / (N_RUNS - 1)
    unit_idx = i % N_UNITS
    unit = f"GR{unit_idx + 1:02d}"

    # Persistent substrate/run-level features.
    feature_12 = clamp(0.615 + 0.040 * season_sin + RNG.normal(0, 0.018), 0.52, 0.72)
    feature_13 = clamp(6.15 + 0.22 * season_cos + RNG.normal(0, 0.10), 5.65, 6.70)

    # Categorical variables: one weakly useful, one intentionally unimportant.
    feature_14 = RNG.choice(["cat_a", "cat_b", "cat_c"], p=[0.45, 0.35, 0.20])
    feature_15 = RNG.choice(["type_x", "type_y", "type_z"], p=[0.34, 0.33, 0.33])
    group_label = RNG.choice(["group_1", "group_2", "group_3"], p=[0.40, 0.35, 0.25])

    # Process latents that become time-series features.
    f1_level = 61.5 + 8.0 * drift + 3.5 * season_sin + RNG.normal(0, 2.2)
    f1_slope = -0.010 + 0.035 * (regime == "regime_a") - 0.012 * (regime == "regime_c")
    f3_level = 28.0 + 1.2 * season_cos + 0.08 * (f1_level - 62) + RNG.normal(0, 0.55)
    f4_level = 54.0 + 12.0 * drift + 3.2 * season_sin + RNG.normal(0, 1.8)
    f5_level = 47.0 + 10.0 * drift + 4.0 * season_cos + RNG.normal(0, 2.0)
    f6_level = 79.0 - 5.0 * drift + 5.0 * season_sin + RNG.normal(0, 2.5)
    f7_level = 15.0 + 9.0 * season_sin + RNG.normal(0, 1.5)
    f8_level = 935.0 + 190.0 * season_cos + 70.0 * (regime == "regime_b") + RNG.normal(0, 45.0)
    f9_level = 0.58 + 0.12 * season_sin - 0.05 * drift + RNG.normal(0, 0.035)
    f10_level = 1.10 + 0.45 * season_sin + 0.35 * drift + RNG.normal(0, 0.08)

    feature_16 = clamp(0.70 + 0.18 * season_cos + RNG.normal(0, 0.08), 0.35, 1.15)
    feature_17 = clamp(feature_16 - 0.08 + 0.13 * (regime == "regime_c") + RNG.normal(0, 0.10), 0.20, 1.25)
    feature_18 = clamp(38.0 + 18.0 * season_sin + 20.0 * (regime == "regime_b") + RNG.normal(0, 8.0), 5.0, 105.0)
    feature_19 = clamp(50.0 + RNG.normal(0, 14.0), 5.0, 95.0)

    # Persistent target structure.
    moisture_score = -2.9 * ((feature_12 - 0.625) / 0.055) ** 2
    feature_13_linear = 0.85 * (feature_13 - 6.10)
    treatment_score = 0.75 * (feature_17 - feature_16)
    external_penalty = -0.018 * feature_18
    unit_effect = [-0.18, 0.08, 0.00, 0.12, -0.08, 0.05][unit_idx]
    cat_effect = {"cat_a": 0.15, "cat_b": -0.03, "cat_c": -0.22}[feature_14]

    # Regime-dependent drivers. These are deliberately different by period.
    f1_centered = (f1_level - 64.0) / 7.5
    f8_optimum = -1.65 * ((f8_level - 1010.0) / 210.0) ** 2
    f4_penalty = -0.105 * max(f4_level - 61.0, 0.0)
    f6_penalty = -0.060 * max(77.0 - f6_level, 0.0)
    f8_f9_interaction = 1.55 * ((f8_level - 900.0) / 260.0) * (f9_level - 0.55)

    if regime == "regime_a":
        regime_component = 1.20 * f1_centered + 0.70 * (f3_level - 28.0)
    elif regime == "regime_b":
        regime_component = f8_optimum + f8_f9_interaction - 0.020 * (feature_18 - 45.0)
    else:
        regime_component = f4_penalty + f6_penalty + 0.95 * (feature_17 - feature_16)

    baseline_drift = 4.8 + 0.8 * season_cos - 1.2 * drift
    random_instability = RNG.normal(0, 0.55)

    target_pct = (
        baseline_drift
        + moisture_score
        + feature_13_linear
        + treatment_score
        + external_penalty
        + unit_effect
        + cat_effect
        + regime_component
        + random_instability
    )
    target_pct = clamp(target_pct, 0.0, 7.5)

    return {
        "run_index": i,
        "run_id": f"RUN_{i + 1:03d}",
        "unit": unit,
        "unit_idx": unit_idx,
        "regime": regime,
        "season_sin": season_sin,
        "season_cos": season_cos,
        "drift": drift,
        "feature_12": feature_12,
        "feature_13": feature_13,
        "feature_14": feature_14,
        "feature_15": feature_15,
        "group_label": group_label,
        "f1_level": f1_level,
        "f1_slope": f1_slope,
        "f3_level": f3_level,
        "f4_level": f4_level,
        "f5_level": f5_level,
        "f6_level": f6_level,
        "f7_level": f7_level,
        "f8_level": f8_level,
        "f9_level": f9_level,
        "f10_level": f10_level,
        "feature_16": feature_16,
        "feature_17": feature_17,
        "feature_18": feature_18,
        "feature_19": feature_19,
        "target_metric_primary": target_pct / 100.0,
        "target_percent": target_pct,
        "contrib_persistent_moisture": moisture_score,
        "contrib_persistent_feature_13": feature_13_linear,
        "contrib_persistent_treatment": treatment_score,
        "contrib_persistent_external": external_penalty,
        "contrib_regime_specific": regime_component,
        "contrib_random_instability": random_instability,
    }


def build_main_timeseries(latents: dict[str, object], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    unit = str(latents["unit"])
    start_ts = start_date + pd.Timedelta(hours=17)
    end_ts = end_date + pd.Timedelta(hours=6)
    ts = pd.date_range(start_ts, end_ts, freq=SAMPLE_FREQ)
    n = len(ts)
    hours = np.arange(n) * 0.25
    days = hours / 24.0
    within_day = 2 * np.pi * (hours % 24.0) / 24.0
    six_hour = 2 * np.pi * hours / 6.0
    ninety_min = 2 * np.pi * hours / 1.5

    roi1 = (hours < 100).astype(float)
    roi2 = ((hours >= 100) & (hours < 200)).astype(float)
    roi3 = (hours >= 200).astype(float)
    unit_idx = int(latents["unit_idx"])
    regime = str(latents["regime"])

    # Per-run eccentricity terms keep traces from looking like cloned sinusoids.
    daily_gain = RNG.uniform(0.15, 1.35)
    duty_gain = RNG.uniform(0.7, 1.6)
    burst_gain = RNG.uniform(0.5, 1.8)
    high_freq_gain = RNG.uniform(0.4, 1.7)

    # feature_1_block: comparatively smooth process state with slow non-periodic
    # drift, weak daily structure, and occasional mild corrections.
    f1 = (
        float(latents["f1_level"])
        + float(latents["f1_slope"]) * hours
        + daily_gain * 0.45 * np.sin(within_day + 0.3 * unit_idx)
        + low_frequency_wander(n, 0.95, knots=8)
        + 1.2 * roi1
        - 1.8 * roi3
        + pulse_train(n, count=2, amplitude=0.65, width_min=18, width_max=52, signed=True)
        + smooth_noise(n, 0.075, persistence=0.985)
    )

    # feature_3: sluggish response signal, intentionally smoother than feature_1.
    f3 = (
        float(latents["f3_level"])
        + 0.035 * pd.Series(f1 - np.nanmean(f1)).rolling(24, min_periods=1).mean().to_numpy()
        + 0.12 * np.cos(within_day + 0.5)
        + 0.55 * roi2
        + low_frequency_wander(n, 0.28, knots=5)
        + smooth_noise(n, 0.035, persistence=0.99)
    )

    # feature_7: external/conditioning-like signal with strong oscillatory
    # character and several frequencies.
    f7 = (
        float(latents["f7_level"])
        + high_freq_gain * 1.5 * np.sin(within_day - 0.8)
        + high_freq_gain * 0.75 * np.sin(six_hour + RNG.uniform(-0.8, 0.8))
        + high_freq_gain * 0.28 * np.sin(ninety_min + RNG.uniform(-0.8, 0.8))
        + 0.10 * days
        + smooth_noise(n, 0.16, persistence=0.76)
    )

    # feature_4_1: duty/cooling-like trace: square-ish cycling, ramps, and bursts.
    duty_cycle = np.sign(np.sin(2 * np.pi * hours / RNG.uniform(7.0, 13.0) + RNG.uniform(-1.0, 1.0)))
    f4 = (
        float(latents["f4_level"])
        + 0.018 * hours
        + 2.0 * roi3
        + duty_gain * 1.25 * duty_cycle
        + 0.55 * np.sin(within_day + 1.1)
        + pulse_train(n, count=5 if regime != "regime_a" else 3, amplitude=1.5, width_min=5, width_max=22)
        + smooth_noise(n, 0.18, persistence=0.84)
    )

    # feature_8: load/air-quality-like signal with broad arcs and occasional
    # spikes, not the same wavelength as the other traces.
    f8 = (
        float(latents["f8_level"])
        + 18.0 * np.sin(2 * np.pi * hours / RNG.uniform(52.0, 110.0) + RNG.uniform(-1.0, 1.0))
        + low_frequency_wander(n, 42.0, knots=6)
        + 30.0 * roi2
        - 25.0 * roi3
        + pulse_train(n, count=4, amplitude=38.0, width_min=2, width_max=10, signed=True)
        + smooth_noise(n, 4.3, persistence=0.72)
    )

    # feature_6: noisy utility/environmental response with bursty disturbances.
    f6 = (
        float(latents["f6_level"])
        - 0.010 * hours
        + daily_gain * 0.75 * np.cos(within_day)
        - 1.6 * roi3
        + pulse_train(n, count=8, amplitude=burst_gain * 0.75, width_min=1, width_max=7, signed=True)
        + RNG.normal(0.0, 0.42 * burst_gain, n)
        + smooth_noise(n, 0.12, persistence=0.68)
    )

    # feature_5: mixed response: partly coupled to feature_4, partly wandering,
    # and with irregular batch-to-batch roughness.
    f5 = (
        float(latents["f5_level"])
        + 0.11 * (f4 - np.nanmean(f4))
        + 0.45 * np.sin(2 * np.pi * hours / RNG.uniform(18.0, 36.0) + 2.2)
        + low_frequency_wander(n, 1.25, knots=10)
        + pulse_train(n, count=3, amplitude=1.1, width_min=9, width_max=35, signed=True)
        + smooth_noise(n, 0.22, persistence=0.90)
    )

    setpoint = np.full(n, 62.0 + 4.0 * float(latents["drift"]) + 1.5 * (latents["regime"] == "regime_c"))

    return pd.DataFrame(
        {
            "t_stamp": ts,
            f"{unit}/ai/{unit}_feature_7/eu": np.clip(f7, 1.0, None),
            f"{unit}/{unit}_feature_1_block/eu": np.clip(f1, 1.0, 99.0),
            f"{unit}/{unit}_feature_1_block/setpoint": setpoint,
            f"{unit}/{unit}_feature_3/eu": np.clip(f3, 1.0, None),
            f"{unit}/ai/{unit}_feature_4_1/eu": np.clip(f4, 1.0, None),
            f"{unit}/ai/{unit}_feature_8/eu": np.clip(f8, 1.0, None),
            f"{unit}/ai/{unit}_feature_6/eu": np.clip(f6, 1.0, None),
            f"{unit}/{unit}_feature_5": np.clip(f5, 1.0, None),
        }
    )


def build_supplemental_timeseries(latents: dict[str, object], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    start_ts = start_date + pd.Timedelta(hours=17)
    end_ts = end_date + pd.Timedelta(hours=6)
    ts = pd.date_range(start_ts, end_ts, freq=SAMPLE_FREQ)
    n = len(ts)
    hours = np.arange(n) * 0.25
    within_day = 2 * np.pi * (hours % 24.0) / 24.0
    roi2 = ((hours >= 100) & (hours < 200)).astype(float)

    f9 = (
        float(latents["f9_level"])
        + 0.025 * np.sin(within_day + 0.3)
        + 0.035 * roi2
        + smooth_noise(n, 0.004)
    )
    f10 = (
        float(latents["f10_level"])
        + 0.060 * np.cos(within_day - 0.7)
        + 0.035 * (hours / hours.max())
        + smooth_noise(n, 0.008)
    )

    return pd.DataFrame(
        {
            "t_stamp": ts,
            "feature_9": np.clip(f9, 0.05, 1.20),
            "feature_10_ext_metric": np.clip(f10, 0.01, None),
        }
    )


def main() -> None:
    TS_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    for folder in (TS_DIR, SUPP_DIR):
        for old_file in folder.glob("*.csv"):
            old_file.unlink()

    metadata_rows: list[dict[str, object]] = []
    treatment_rows: list[dict[str, object]] = []
    external_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []

    base_date = pd.Timestamp("2025-01-06")

    for i in range(N_RUNS):
        latents = build_run_latents(i)
        run_id = str(latents["run_id"])
        unit = str(latents["unit"])
        start_date = base_date + pd.Timedelta(days=3 * i)
        end_date = start_date + pd.Timedelta(days=14)

        main_ts = build_main_timeseries(latents, start_date, end_date)
        supp_ts = build_supplemental_timeseries(latents, start_date, end_date)

        main_ts.to_csv(TS_DIR / f"{run_id}.csv", index=False)
        supp_ts.to_csv(SUPP_DIR / f"{run_id}_{unit}_supplemental.csv", index=False)

        metadata_rows.append(
            {
                "source_run_id": run_id,
                "source_run_start_date": start_date.date().isoformat(),
                "source_run_end_date": end_date.date().isoformat(),
                "source_unit_label": unit,
                "source_group_label": latents["group_label"],
                "source_feature_12": round(float(latents["feature_12"]), 5),
                "source_feature_13": round(float(latents["feature_13"]), 5),
                "source_feature_14": latents["feature_14"],
                "source_feature_15": latents["feature_15"],
                "source_target_metric": round(float(latents["target_metric_primary"]), 5),
            }
        )

        treatment_rows.append(
            {
                "source_run_id": run_id,
                "feature_16": round(float(latents["feature_16"]), 5),
                "feature_17": round(float(latents["feature_17"]), 5),
            }
        )

        external_rows.extend(
            [
                {
                    "source_run_id": run_id,
                    "source_category": "category_group_a",
                    "source_status": "status_group_a",
                    "source_metric_a": round(float(latents["feature_18"]) + RNG.normal(0, 3.0), 3),
                    "source_metric_b": round(float(latents["feature_19"]) + RNG.normal(0, 5.0), 3),
                },
                {
                    "source_run_id": run_id,
                    "source_category": "category_group_b",
                    "source_status": "status_group_b",
                    "source_metric_a": round(float(latents["feature_18"]) + RNG.normal(0, 4.0), 3),
                    "source_metric_b": round(float(latents["feature_19"]) + RNG.normal(0, 5.0), 3),
                },
                {
                    "source_run_id": run_id,
                    "source_category": "category_group_c",
                    "source_status": "status_group_c",
                    "source_metric_a": round(float(latents["feature_18"]) + RNG.normal(0, 4.0), 3),
                    "source_metric_b": round(float(latents["feature_19"]) + RNG.normal(0, 5.0), 3),
                },
            ]
        )

        truth_rows.append(
            {
                "run_id": run_id,
                "run_start_date": start_date.date().isoformat(),
                "unit_label": unit,
                "synthetic_regime": latents["regime"],
                "season_sin": round(float(latents["season_sin"]), 6),
                "season_cos": round(float(latents["season_cos"]), 6),
                "drift": round(float(latents["drift"]), 6),
                "latent_feature_1_level": round(float(latents["f1_level"]), 5),
                "latent_feature_4_1_level": round(float(latents["f4_level"]), 5),
                "latent_feature_8_level": round(float(latents["f8_level"]), 5),
                "latent_feature_9_level": round(float(latents["f9_level"]), 5),
                "target_metric_primary": round(float(latents["target_metric_primary"]), 5),
                "target_percent": round(float(latents["target_percent"]), 3),
                "contrib_persistent_moisture": round(float(latents["contrib_persistent_moisture"]), 5),
                "contrib_persistent_feature_13": round(float(latents["contrib_persistent_feature_13"]), 5),
                "contrib_persistent_treatment": round(float(latents["contrib_persistent_treatment"]), 5),
                "contrib_persistent_external": round(float(latents["contrib_persistent_external"]), 5),
                "contrib_regime_specific": round(float(latents["contrib_regime_specific"]), 5),
                "contrib_random_instability": round(float(latents["contrib_random_instability"]), 5),
            }
        )

    pd.DataFrame(metadata_rows).to_csv(INPUT_DIR / "run_metadata.csv", index=False)
    pd.DataFrame(treatment_rows).to_csv(INPUT_DIR / "run_treatment_metrics.csv", index=False)
    pd.DataFrame(external_rows).to_csv(INPUT_DIR / "external_run_metrics.csv", index=False)
    pd.DataFrame(truth_rows).to_csv(INPUT_DIR / "synthetic_ground_truth.csv", index=False)

    print(f"Generated {N_RUNS} synthetic runs")
    print(f"Main time-series files: {TS_DIR}")
    print(f"Supplemental time-series files: {SUPP_DIR}")
    print(f"Run metadata: {INPUT_DIR / 'run_metadata.csv'}")
    print(f"Synthetic ground truth: {INPUT_DIR / 'synthetic_ground_truth.csv'}")


if __name__ == "__main__":
    main()
