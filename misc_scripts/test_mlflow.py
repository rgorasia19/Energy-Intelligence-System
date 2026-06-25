import mlflow
import numpy as np
import os

def main():
    # -----------------------------
    # CONFIG
    # -----------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlflow_sanity_check")

    print("Starting MLflow test run...")

    # -----------------------------
    # START RUN
    # -----------------------------
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run started: {run_id}")

        # -----------------------------
        # LOG PARAMS
        # -----------------------------
        mlflow.log_param("test_param", 42)
        mlflow.log_param("model", "sanity_check_model")

        # -----------------------------
        # SIMULATE TRAINING LOOP
        # -----------------------------
        for epoch in range(10):
            train_loss = np.exp(-epoch / 5) + np.random.normal(0, 0.01)
            val_loss = np.exp(-epoch / 4) + np.random.normal(0, 0.01)
            entropy = np.clip(np.random.rand(), 0, 1)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("entropy", entropy, step=epoch)

            print(f"epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # -----------------------------
        # LOG ARTIFACT (dummy file)
        # -----------------------------
        artifact_path = "dummy_artifact.txt"
        with open(artifact_path, "w") as f:
            f.write("MLflow test artifact OK\n")

        mlflow.log_artifact(artifact_path)

        # -----------------------------
        # LOG SIMPLE MODEL (placeholder)
        # -----------------------------
        dummy_model_path = "dummy_model.txt"
        with open(dummy_model_path, "w") as f:
            f.write("this is a fake model checkpoint\n")

        mlflow.log_artifact(dummy_model_path)

        print("Run completed successfully.")

    print("MLflow test finished. Check UI.")

if __name__ == "__main__":
    main()