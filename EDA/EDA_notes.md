# Exploratory Data Analysis
## Time Series Observations:
By plotting the time series data for random days for different power stations, we can observe the following:

1. Demand picks up at key times during the day, primarily between 4-6AM and 4-8PM. 
2. We note that there are some substations, that have some noise in their demand curves, these are likely areas with a mix of residential and industrial load. 
3. Some substations show negative load or drop midday during the summer, indicating the presence of local solar generation exporting back into the grid.

## PCA Analysis
Key Results
1. Dominant Global Mode (PC1)
All substations have similar positive loadings in PC1
Indicates strong shared structure across the network

Interpretation:

PC1 represents the overall system demand level.

This implies:

Substations rise and fall together
Aggregation into a total load signal is valid

2. Secondary Mode (PC2) — Structural Variation
Mixed positive and negative loadings
Separates substations based on differing behaviour

Interpretation:

PC2 captures structural differences in load behaviour.

Likely drivers:

Distributed solar generation (midday export)
Industrial vs residential demand profiles
Localised generation/load imbalance

3. Higher Components
Lower explained variance
No consistent structure across substations

### Interpretation:

Higher PCs capture noise and localised effects.

PCA Projection (PC1 vs PC2)

Projection of time steps into PCA space reveals:

- A continuous cloud, not discrete clusters
- A diagonal trend → system behaviour evolves with demand level
- Two broad regimes:
- High PC2 → normal consumption-dominated state
- Low PC2 → altered state (likely solar-influenced/export conditions)

### Key Insights
Low-Dimensional Structure

The system is largely governed by:

- 1 dominant global mode (PC1)
- 1–2 secondary variation modes (PC2, possibly PC3)

This suggests:

- The grid is highly structured
- Behaviour can be approximated with a small number of components
- Demand is Not Sufficient Alone

Same demand level (PC1) can correspond to different system states (PC2).

### Implication:

Demand forecasting alone is insufficient — structural state must also be considered.

### Evidence of Distributed Generation

Negative flows observed during midday (summer months) align with:

Low PC2 values
Distinct PCA regime

### Interpretation:

PCA captures the influence of embedded solar generation on system behaviour.

### Implications for Modelling
Forecasting
Model should include:
PC1 (magnitude of demand)
PC2 (system structure)