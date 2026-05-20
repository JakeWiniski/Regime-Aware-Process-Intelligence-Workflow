# Regime-Aware Process Intelligence

This repository implements a **Regime-Aware Process Intelligence (RAPI)** workflow for analyzing, monitoring, and reasoning about complex, non-stationary processes using interpretable statistical and machine-learning tools.

The core idea is that in evolving biological and cyber-physical systems, *change itself is informative*. Rather than assuming stability and enforcing fixed control structures prematurely, this workflow treats models, relationships, and operating assumptions as **provisional hypotheses** that must be continuously tested against reality.

---

## Repository Structure

This workflow is implemented as **two complementary Jupyter notebooks**:

### 1. `Rolling_Surrogate_Regime_Analysis.ipynb`  
Detects and characterizes **local regime shifts** over time.

### 2. `Global_Model.ipynb`  
Builds a **global integrative model** to contextualize and interpret those regimes across the full dataset.

Together, they provide a multi-scale view of system behavior:

| Perspective | Notebook | Role |
|------------|-----------|-------|
Local / temporal | Rolling surrogate | Detect when and how the system changes |
Global / structural | Global model | Understand what the system is, in aggregate |

---

## Conceptual Motivation

Many real-world biological and industrial processes are:

- non-stationary,
- only partially observable,
- under-characterized,
- and evolving while operating at scale.

In such systems:

- growth phases shift,
- metabolic or structural behavior reorganizes,
- stress responses accumulate and release,
- and empirical models lose validity over time.

This violates the assumptions of classical control and static modeling.

The framework therefore:

- treats every run as both production and experiment,
- treats models as temporary instruments, not truths,
- and emphasizes **detecting drift and regime change rather than hiding it**.

---

## Notebook 1: Rolling Surrogate Regime Analysis

The rolling notebook implements a **sliding-window surrogate modeling approach**:

1. Raw run-level time series are aligned and segmented into user-defined temporal windows.
2. Compact features (trends, variability, autocorrelation, frequency content, coupling metrics) are computed per run.
3. A simple, interpretable model is repeatedly trained on a rolling window of recent runs.
4. Changes in:
   - model performance,
   - residual structure,
   - and feature importance  
   are monitored as **signals of regime change**.

The model is not treated as a predictor to be optimized, but as a **measurement device** for detecting structural drift.

---

## Notebook 2: Global Integrative Model

The global notebook operates on the outputs of the rolling analysis to:

- integrate all features into a unified representation,
- reduce dimensionality to reveal dominant modes of variation,
- fit a stable global model across all regimes,
- and contextualize local regimes within the full system structure.

This supports:

- comparison between regimes,
- identification of persistent drivers,
- detection of structurally distinct or risky system states,
- and long-horizon reasoning about system behavior.

---

## Design Principles

This workflow is built around several core principles:

- **Regime awareness over stationarity:** assume the system changes unless proven otherwise.
- **Explainability over optimization:** models are tools for understanding, not just prediction.
- **Change as signal:** drift, instability, and model failure are primary observables.
- **Human-in-the-loop:** outputs are designed for interpretation and decision support, not autonomous control.
- **Safe learning at scale:** production and learning occur simultaneously.

---

## What This Enables

This workflow enables teams to:

- operate complex systems before full stabilization,
- detect regime shifts before catastrophic failure,
- preserve learning velocity while reducing operational risk,
- gradually converge toward validated control where appropriate,
- and maintain situational awareness in under-characterized domains.

Rather than replacing expert judgment, it augments it — providing structured signals that help humans reason about complex, evolving systems under uncertainty.
