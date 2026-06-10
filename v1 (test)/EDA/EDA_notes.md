# Exploratory Data Analysis
## Time Series Observations:
By plotting the time series data for random days for different power stations, we can observe the following:

1. Demand picks up at key times during the day, primarily between 4-6AM and 4-8PM. 
2. We note that there are some substations, that have some noise in their demand curves, these are likely areas with a mix of residential and industrial load. 
3. Some substations show negative load or drop midday during the summer, indicating the presence of local solar generation exporting back into the grid.

## Correlation Analysis
Global Network Patterns: Analysis of the heavy system-wide positive correlations ($0.65 \le r \le 0.98$) driven by shared macro-level variables like weather and consumer behavior cycles, and the resulting risks of coincident peaks. In some cases, due to negative generation from local solar installations, we see some instances of negative correlation between substations.  
Margam is an outlier wrt to other stations, its correlation with other stations is very low. This could be due to Margam being situated in Port Talbot, an industrial hub, with different load characteristics than other stations. 

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

## Autocorrelation
- High-Frequency Diurnal Rhythm (The "Spiky" Band): The dense, jagged band represents strong daily (24-hour) seasonality. The sharp upward spikes at regular intervals show that grid demand at any given hour remains heavily correlated with the exact same hour on preceding days.
- Macro Seasonal Waves: The underlying sinusoidal drift across the X-axis reveals the broader seasonal cycle of the grid.The initial decay transitions into a negative correlation trough (around lags 5,000–9,000), showing the inverse relationship between opposing seasons (e.g., winter peaks vs. summer or spring baseloads). The curve rises back into a secondary positive hump (around lags 12,000–15,000), capturing the structural return to the same season one full cycle later.
- Sampling Interval Clue: Because a standard year contains 8,760 hours, the fact that the macro cycle peaks and resolves over a horizon extending up to 16,000+ points strongly points to half-hourly data resolution (where a full year comprises 17,520 settlement periods), which is highly typical for UK electrical grid reporting.
- Modeling Impact: The notes outline why standard ARIMA will struggle here, confirming the need for frameworks that support multiple seasonalities (such as SARIMA or Prophet) or explicit lag-feature engineering ($t-48$ half-hours, $t-17,520$ half-hours) in machine learning architectures.