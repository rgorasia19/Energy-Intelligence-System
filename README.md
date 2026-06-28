# Energy-Intelligence-System

This repository contains the evolution of the Energy Intelligence System, structured across different versions that track the progression of our exploratory analysis and predictive modeling strategies.

## Versions Overview

### v1 (Exploratory Data Analysis)
Focuses on understanding the time-series data. Includes PCA analysis to identify the dominant structural states of the grid, autocorrelation studies to detect high-frequency diurnal rhythms and macro seasonal waves, and correlation analysis of substation behaviours.

### v2 (XGBoost Baselines)
Early baseline models using XGBoost for different forecast horizons (e.g., 1 day to 1 week). Demonstrated strong initial $R^2$ scores (~0.8 to 0.99) but highlighted overfitting challenges, establishing a benchmark for future deep learning approaches.

### v3 (GMM Regime Analysis & Neural HMM)
Explored modelling grid dynamics using discrete states. Applied Gaussian Mixture Models (GMM) on PCA features to interpret 4 core grid "regimes" (e.g., high generation vs. high import). Also includes initial experiments with a Neural Hidden Markov Model, which suffered from regime collapse (overfitting to a single state).

### v4 (Temporal Mixture of Experts - Design)
A pivot from HMMs to a Temporal Mixture of Experts (MoE) architecture. Designed a system with a shared encoder and a gating network to distribute predictions across specialized experts: a Linear model (for trends), a Fourier Transform model (for periodicities), and an Attention-based model (for non-linearities).

### v5 (Unified Regime Model Implementation)
Implementation of the Attention-based Mixture of Experts predictor (`UnifiedRegimeModel`). Uses specialized MLP expert networks trained with a gating mechanism to adapt to different grid states.

### v6 (Continuous State Forecaster)
Transition from discrete MoE regimes to an End-to-End Continuous Latent State Space model. Introduces the `ContinuousStateForecaster`, employing a Transformer backbone to model continuous, fluid transitions in the underlying state of the power grid.