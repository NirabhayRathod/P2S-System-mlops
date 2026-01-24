import os
import sys
import pickle
import yaml
import numpy as np

import mlflow
import mlflow.pyfunc

from sklearn.metrics import roc_auc_score, mean_squared_error
from logger import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Helper: Required environment variables

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


# MAIN EVALUATION FUNCTION

def evaluate_models():

    # ---------------- Load config ----------------
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION USING MLFLOW (CI/CD)")
    print("=" * 60)

    # ---------------- Load test data ----------------
    try:
        with open(paths["test_data"], "rb") as f:
            test_data = pickle.load(f)

        X_test_pwave = test_data["X_test_pwave"]
        y_test_pwave = test_data["y_test_pwave"]
        X_test_swave = test_data["X_test_swave"]
        y_test_swave = test_data["y_test_swave"]

    except Exception as e:
        print(f" Failed to load test data: {e}")
        return 1

    # ---------------- MLflow + DagsHub setup ----------------
    try:
        DAGSHUB_USER = require_env("DAGSHUB_USER")
        DAGSHUB_TOKEN = require_env("DAGSHUB_TOKEN")
        DAGSHUB_MLFLOW_URI = require_env("DAGSHUB_MLFLOW_URI")
    except RuntimeError as e:
        print(f" Environment error: {e}")
        return 1

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    # EVALUATE P-WAVE CLASSIFIER (MLFLOW)
    
    print("\n Evaluating P-wave Classifier (Production model)...")
    try:
        pwave_model = mlflow.pyfunc.load_model(
            "models:/P2S_PWAVE_MODEL/Production"
        )

        y_pred_proba = pwave_model.predict(X_test_pwave)

        pwave_roc_auc = roc_auc_score(y_test_pwave, y_pred_proba)

        print(f"   ROC-AUC Score: {pwave_roc_auc:.4f}")
        print(
            f"   Threshold: >0.80 "
            f"{'PASSED ' if pwave_roc_auc > 0.80 else 'FAILED '}"
        )

        if pwave_roc_auc <= 0.80:
            print(f" FAIL: ROC-AUC {pwave_roc_auc:.4f} <= 0.80")
            return 1

    except Exception as e:
        print(f" P-wave evaluation failed: {e}")
        return 1

    # EVALUATE S-WAVE REGRESSOR (MLFLOW)
    
    print("\n🔍 Evaluating S-wave Regressor (Production model)...")
    try:
        swave_model = mlflow.pyfunc.load_model(
            "models:/P2S_SWAVE_MODEL/Production"
        )

        y_pred = swave_model.predict(X_test_swave)

        ss_res = np.sum((y_test_swave - y_pred) ** 2)
        ss_tot = np.sum((y_test_swave - np.mean(y_test_swave)) ** 2)
        swave_r2 = 1 - (ss_res / ss_tot)

        rmse = np.sqrt(mean_squared_error(y_test_swave, y_pred))

        print(f"   R2 Score: {swave_r2:.4f}")
        print(f"   RMSE: {rmse:.2f} seconds")
        print(
            f"   Threshold: R² >0.80 "
            f"{'PASSED ' if swave_r2 > 0.80 else 'FAILED '}"
        )

        if swave_r2 <= 0.80:
            print(f" FAIL: R2 {swave_r2:.4f} <= 0.80")
            return 1

    except Exception as e:
        print(f" S-wave evaluation failed: {e}")
        return 1

    # ================= SUCCESS =================
    print("\n" + "=" * 60)
    print(" ALL MODELS PASSED QUALITY GATES")
    print("=" * 60)
    print(f"P-wave ROC-AUC: {pwave_roc_auc:.4f} (>0.80)")
    print(f"S-wave R2: {swave_r2:.4f} (>0.80), RMSE: {rmse:.2f}s")
    print("\n Ready for deployment!")

    return 0


# ENTRY POINT (CI/CD)
def main():
    logging.info("Starting MLflow-based model evaluation for CI/CD...")
    exit_code = evaluate_models()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
