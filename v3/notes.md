# NOTES
## PCA Findings
Features were batched according to the following:
- INTERCONNECT_FLOW : 10 components (95%)
- GENERATION_FLOW : 7 components (95%)
- CAPACITY_FLOW : 7 components (95%)

The following variance ratios were found:
- INTERCONNECT_FLOW : [0.33452991 0.15862498 0.11369631 0.11013582 0.07698258 0.06033829
 0.04866917 0.041906   0.02951065]
- GENERATION_FLOW : [0.34131531 0.18208316 0.15800129 0.10723764 0.07406673 0.05993148
 0.04113928]
- CAPACITY_FLOW : [0.24441656 0.23221963 0.13220267 0.12182211 0.11235142 0.06931471
 0.04895699]

This suggests that interconnect flows a strong global state axis, generation has a dominant axis + clear secondary structure and capacity is more multi-dimensional and flexible.

## GMM Regime Analysis
A Gaussian Mixture Model (4 components, full covariance) was fitted on the PCA features. 
- **Persistence**: High (~97.5%), indicating the regimes capture stable macroeconomic or seasonal states rather than hourly noise.
- **Balance**: The dataset is well-distributed among the four regimes.

### Regime Interpretations
* **Regime 1 (30.8%) - "Self-Sufficient / High Generation"**: Driven by extremely high domestic generation (`GEN_PC0`) and primary capacity (`CAP_PC0`), with very low reliance on primary interconnect flows (`INTER_PC0`). 
* **Regime 0 (29.9%) - "Secondary Balancing"**: Primary generation is relatively average. This state is defined by high secondary interconnect flows (`INTER_PC1`), representing a balancing act between a subset of interconnector lines.
* **Regime 2 (19.9%) - "High Import / Low Generation"**: The mirror opposite of Regime 1. High primary interconnect flow (`INTER_PC0`) and capacity utilization (`CAP_PC1`), but severely low domestic generation (`GEN_PC0`).
* **Regime 3 (19.4%) - "Niche / Transition State"**: A smaller regime characterized by below-average generation and primary interconnect usage, instead relying on specific, lower-variance capacity lines (`CAP_PC2`).
