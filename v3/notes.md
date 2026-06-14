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

## Final Dataset
