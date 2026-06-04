import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, PrecisionRecallDisplay

def main():
    parser = argparse.ArgumentParser(description="Train or infer with the regime classifier.")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="infer",
                        help="Mode to run the script in: 'train' or 'infer'")
    args = parser.parse_args()

    #1. Load Data
    pc_df = pd.read_csv("pc_df.csv")
    pc_df.index = pd.to_datetime(pc_df["Timestamp"])

    df = pd.read_csv("../data_lake/master_df.csv")
    df.index = pd.to_datetime(df["Timestamp"])

    #2. Define Target
    y = pc_df["regime"].shift(-1).dropna()

    #3. Feature Engineering
    # Time Features
    x = pd.DataFrame(index=df.index)
    x["hour"] = df.index.hour
    x["dayofweek"] = df.index.dayofweek
    x["month"] = df.index.month
    x["hour_sin"] = np.sin(2*np.pi*x["hour"]/24)
    x["hour_cos"] = np.cos(2*np.pi*x["hour"]/24)

    #Lags and Rollings
    for i in range(5):
        x[f"load_lag_{i+1}"] = df["Total"].shift((i+1)*48)
        x[f"pc1_lag_{i+1}"] = pc_df["PC1"].shift((i+1)*48)
        x[f"rolling_mean_{i+1}"] = df["Total"].rolling((i+1)*48).mean()
        x[f"rolling_std_{i+1}"] = df["Total"].rolling((i+1)*48).std()

    #4. Combine y and X
    model_df = x.copy()
    model_df["Target"] = y
    model_df = model_df.dropna()

    x = model_df.drop(columns=["Target"])
    y = model_df["Target"]

    #5. Train-Test Split
    split = int(len(df)*0.8)

    train_x = x[:split]
    train_y = y[:split]

    test_x = x[split:]
    test_y = y[split:]

    if args.mode == "train":
        print("Training model...")
        #6. Scale X data
        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        test_x_scaled = scaler.transform(test_x)

        #7. Train RF
        model = RandomForestClassifier(
            n_estimators = 500, 
            max_depth = 10,
            class_weight="balanced",
            random_state=42
        )

        model.fit(train_x_scaled, train_y)
        joblib.dump({
            "model" : model,
            "scaler" : scaler},
            "regime_model.pkl")
        print("Model trained and saved to regime_model.pkl")
        
        y_pred = model.predict(test_x_scaled)
        
    elif args.mode == "infer":
        print("Loading model for inference...")
        try:
            data = joblib.load("regime_model.pkl")
            model = data["model"]
            scaler = data["scaler"]
        except FileNotFoundError:
            print("Model file not found. Please run with '--mode train' first.")
            return

        test_x_scaled = scaler.transform(test_x)
        y_pred = model.predict(test_x_scaled)
        print("Inference complete.")

    print(classification_report(test_y, y_pred))

    conf_matrix = confusion_matrix(test_y, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.jpg", dpi=600)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_y.index,test_y.values, label="true")
    plt.plot(test_y.index,y_pred, alpha=0.7, label="pred")
    plt.legend()
    plt.savefig("results.jpg", dpi=600)
    plt.show()

    PrecisionRecallDisplay.from_estimator(
    model,
    test_x_scaled,
    test_y,
    name="RandomForestClassifier")
    plt.title("Precision-Recall Curve")
    plt.savefig("PR_curve.jpg", dpi=600)
    plt.show()






if __name__ == "__main__":
    main()
