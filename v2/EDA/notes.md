# EDA FINDINGS
## 1. Global Correlations : 
Using the pearsonr function from scipy.stats we find the following correlations : 
- SOLAR X CARBON :  -0.3594161772455381
- WIND X CARBON :  -0.704717867460791
- HYDRO X Carbon :  0.04309209045361147

## 2. National Demand and Generation Mix
The following charts display the trends over time for National Demand and the overall Generation Mix.

![National Demand over time](c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/ND.jpg)
![Generation Mix over time](c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/gen_mix.jpg)

## 3. Seasonality
The seasonality analysis looks at the average values of National Demand (ND), Solar Generation, and Carbon Intensity across different time features: Hour of the Day, Day of the Week, Week of the Year, and Month.

![Seasonality](c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/seasonality.jpg)

## 4. Principal Component Analysis (PCA)
PCA was performed to understand the variance and main drivers within the dataset.
- **PCA Loadings**: The components loadings are saved in [PCA_loadings.csv](file:///c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/PCA_loadings.csv).
- **Explained Variance**: The cumulative explained variance by the PCA components shows how many components are needed to retain the majority of the data's variance.

![PCA Explained Variance](c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/PCA_Explained_Variance.jpg)
- **Scatter Plot**: The scatter plot visualizes the first two principal components.

![PCA Scatter Plot](c:/Users/ronak/OneDrive/Desktop/Random/Energy-Intelligence-System/v2/EDA/PCA_Scatter_Plot.jpg)
