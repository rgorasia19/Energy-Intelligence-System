import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pc_df = pd.read_csv("pc_df.csv")
pc_df.index = pd.to_datetime(pc_df["Timestamp"])
pc_df.drop(columns=["Timestamp"], inplace=True)

master_df = pd.read_csv("../data_lake/master_df.csv")
master_df.index = pd.to_datetime(master_df["Timestamp"])
master_df.drop(columns=["Timestamp"], inplace=True)

def average_pc2_by_time_of_day(df):
    """
    Averages PC2 by time of day and plots the result.
    Assumes df has a DatetimeIndex and a column named 'PC2' or 'pc2'.
    """
    col_name = 'PC2' if 'PC2' in df.columns else 'pc2'
    if col_name not in df.columns:
        print(f"Error: Could not find 'pc2' or 'PC2' column in the dataframe.")
        return
    
    # Calculate average by time of day
    avg_by_time = df.groupby(df.index.time)[col_name].mean()
    return avg_by_time

def average_pc2_by_month(df):
  """
  Average PC2 by month and plots the results
  """
  col_name = 'PC2' if 'PC2' in df.columns else 'pc2'
  if col_name not in df.columns:
      print(f"Error: Could not find 'pc2' or 'PC2' column in the dataframe")
      return
  
  avg_by_month = df.groupby(df.index.month)[col_name].mean()
  return avg_by_month

def seasonality_plots():
  time = average_pc2_by_time_of_day(pc_df)
  month = average_pc2_by_month(pc_df)

  plt.figure(figsize=(12, 14))
  plt.subplot(2, 1, 1)
  time.plot()
  plt.title(f"Average PC2 by Time of Day")
  plt.xlabel("Time of Day")
  plt.ylabel(f"Average PC2")
  plt.grid(True)

  plt.subplot(2, 1, 2)
  month.plot()
  plt.title(f"Average PC2 by Month")
  plt.xlabel("Month")
  plt.ylabel(f"Average PC2")
  plt.grid(True)

  plt.tight_layout()
  plt.savefig("pc2_seasonality.jpg", dpi=600)
  plt.show()

def correlations(pc_df, master_df,label=None):
  indv_master_cols = [col for col in master_df.columns if col != "Total"]
  master_df["Mean_Indiv"] = master_df[indv_master_cols].mean(axis=1)
  master_df["Std_Indiv"] = master_df[indv_master_cols].std(axis=1)

  mean_indiv_corr = master_df["Mean_Indiv"].corr(pc_df["PC2"])
  std_indiv_corr = master_df["Std_Indiv"].corr(pc_df["PC2"])


  plt.figure(figsize=(12, 12))
  plt.subplot(2, 1, 1)
  plt.scatter(master_df["Mean_Indiv"], pc_df["PC2"],alpha=0.1)
  if label:
    plt.title(f"Correlation between Mean_Indiv and PC2 for {label} quantile")
  else:
    plt.title(f"Correlation between Mean_Indiv and PC2")
  plt.xlabel("Mean_Indiv")
  plt.ylabel("PC2")

  plt.subplot(2, 1, 2)
  plt.scatter(master_df["Std_Indiv"], pc_df["PC2"],alpha=0.1)
  if label:
    plt.title(f"Correlation between Std_Indiv and PC2 for {label} quantile")
  else:
    plt.title(f"Correlation between Std_Indiv and PC2")
  plt.xlabel("Std_Indiv")
  plt.xlabel("Std_Indiv")
  plt.ylabel("PC2")
  plt.tight_layout()
  if label:
    plt.savefig(f"mean+std_indiv_corr_{label}.jpg", dpi=600)
  else:
    plt.savefig("mean+std_indiv_corr.jpg", dpi=600)
  plt.show()

  print("Correlation between Mean_Indiv and PC2:", mean_indiv_corr)
  print("Correlation between Std_Indiv and PC2:", std_indiv_corr)

def pc2_binning_and_correlations(pc_df, master_df):
  indv_master_cols = [col for col in master_df.columns if col != "Total"]
  lower_pc2_quantile = pc_df["PC2"].quantile(0.25)
  middle_pc2_quantile = pc_df["PC2"].quantile(0.5)
  upper_pc2_quantile = pc_df["PC2"].quantile(0.75)

  lower_pc2_index = pc_df[pc_df["PC2"] < lower_pc2_quantile].index
  middle_pc2_index = pc_df[(pc_df["PC2"] >= lower_pc2_quantile) & (pc_df["PC2"] < middle_pc2_quantile)].index
  upper_pc2_index = pc_df[pc_df["PC2"] >= middle_pc2_quantile].index

  lower_master_df = master_df.loc[lower_pc2_index]
  middle_master_df = master_df.loc[middle_pc2_index]
  upper_master_df = master_df.loc[upper_pc2_index]

  print("\n----LOWER QUANTILE RESULTS----")
  correlations(pc_df.loc[lower_pc2_index], lower_master_df, "lower")
  print("\n----MIDDLE QUANTILE RESULTS----")
  correlations(pc_df.loc[middle_pc2_index], middle_master_df, "middle")
  print("\n----UPPER QUANTILE RESULTS----")
  correlations(pc_df.loc[upper_pc2_index], upper_master_df, "upper")  

if __name__ == "__main__":
  correlations(pc_df, master_df)
    
