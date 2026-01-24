import os
import sys
import yaml
import pickle
import logging
import time

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.feature_engineer import SeismicFeatureEngineer
from src.model_pwave import PWaveClassifier
from src.model_swave import SWaveRegressor

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# PYFUNC WRAPPERS (CRITICAL FOR ARTIFACT CREATION)
# ------------------------------------------------------------------
import pandas as pd

import numpy as np
import pandas as pd
import mlflow.pyfunc


class PWavePyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            X = model_input.to_numpy()
        elif isinstance(model_input, np.ndarray):
            X = model_input
        else:
            X = np.array(model_input)

        return self.model.predict(X)


class SWavePyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            X = model_input.to_numpy()
        elif isinstance(model_input, np.ndarray):
            X = model_input
        else:
            X = np.array(model_input)

        return self.model.predict(X)



def train_earthquake_models():

    # ---------------- Load Params ----------------
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]

    # ---------------- MLflow + DagsHub Setup ----------------
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USER")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    mlflow.set_tracking_uri(
        os.getenv('MLFLOW_TRACKING_URI')
    )
    mlflow.set_experiment("earthquake_warning_system")

    client = MlflowClient()
    logging.info("Connected to DagsHub MLflow")


    engineer = SeismicFeatureEngineer()
    X, y_pwave, y_swave, features = engineer.run_pipeline(
        paths["data_raw"],
        paths["data_processed"]
    )

    # ---------------- Train / Val / Test Split ----------------
    X_train, X_temp, y_p_train, y_p_temp = train_test_split(
        X, y_pwave, test_size=0.3, random_state=42
    )
    X_val, X_test, y_p_val, y_p_test = train_test_split(
        X_temp, y_p_temp, test_size=0.5, random_state=42
    )

    X_train_s, X_temp_s, y_s_train, y_s_temp = train_test_split(
        X, y_swave, test_size=0.3, random_state=42
    )
    X_val_s, X_test_s, y_s_val, y_s_test = train_test_split(
        X_temp_s, y_s_temp, test_size=0.5, random_state=42
    )

    # ---------------- Save Test Data ----------------
    os.makedirs(os.path.dirname(paths["test_data"]), exist_ok=True)
    with open(paths["test_data"], "wb") as f:
        pickle.dump(
            {
                "X_test_pwave": X_test,
                "y_test_pwave": y_p_test,
                "X_test_swave": X_test_s,
                "y_test_swave": y_s_test,
            },
            f
        )

    # ======================================================
    # TRAIN P-WAVE CLASSIFIER
    # ======================================================
    with mlflow.start_run(run_name="pwave_classifier") as run:

        classifier = PWaveClassifier()
        classifier.build_models()

        results = classifier.train_all(
            X_train, y_p_train,
            X_val, y_p_val
        )

        best_name, best_model = classifier.select_best_model()

        mlflow.log_param("best_pwave_model", best_name)
        mlflow.log_metric(
            "best_pwave_roc_auc",
            results[best_name]["metrics"]["roc_auc"]
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PWavePyFuncModel(best_model),
            input_example=X_train[:5],
            registered_model_name="P2S_PWAVE_MODEL"
        )

    time.sleep(5)

    pwave_version = client.get_latest_versions(
        "P2S_PWAVE_MODEL", stages=["None"]
    )[0]

    client.transition_model_version_stage(
        name="P2S_PWAVE_MODEL",
        version=pwave_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    logging.info("P-Wave model promoted to Production")

    # ======================================================
    # TRAIN S-WAVE REGRESSOR
    # ======================================================
    with mlflow.start_run(run_name="swave_regressor") as run:

        regressor = SWaveRegressor()
        regressor.build_models()

        results = regressor.train_all(
            X_train_s, y_s_train,
            X_val_s, y_s_val
        )

        best_name, best_model = regressor.select_best_model(metric="rmse")

        mlflow.log_param("best_swave_model", best_name)
        mlflow.log_metric(
            "best_swave_rmse",
            results[best_name]["metrics"]["rmse"]
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SWavePyFuncModel(best_model),
            input_example=X_train_s[:5],
            registered_model_name="P2S_SWAVE_MODEL"
        )

    time.sleep(5)

    swave_version = client.get_latest_versions(
        "P2S_SWAVE_MODEL", stages=["None"]
    )[0]

    client.transition_model_version_stage(
        name="P2S_SWAVE_MODEL",
        version=swave_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    logging.info("S-Wave model promoted to Production")

    print("\n TRAINING COMPLETE")
    print(" Artifacts created in DagsHub")
    print(" Models registered & promoted to Production")

# ------------------------------------------------------------------
if __name__ == "__main__":
    train_earthquake_models()
