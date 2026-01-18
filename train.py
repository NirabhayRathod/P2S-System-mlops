import mlflow
import yaml , os , pickle
from sklearn.model_selection import train_test_split
from src.feature_engineer import SeismicFeatureEngineer
from src.model_pwave import PWaveClassifier
from src.model_swave import SWaveRegressor
from logger import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def train_earthquake_models():
    with open('params.yaml','r')as f:
        config=yaml.safe_dump_all(f)
    paths=config['paths']
    
    # Set MLflow tracking URI (DagsHub)
    logging.info('mlfow setup done')
    mlflow.set_tracking_uri("https://dagshub.com/yourname/P2S-System-mlops.mlflow")
    mlflow.set_experiment("earthquake_warning_system")
    logging.info('mlfow setup done with named - earthquake_warning_system - ')
    
    # Load 
    engineer = SeismicFeatureEngineer()
    X, y_pwave, y_swave, features = engineer.run_pipeline(
        paths['data_raw'], paths['data_processed']
    )
    
       # Split for P-wave classification (70/15/15)
    X_train, X_temp, y_pwave_train, y_pwave_temp = train_test_split(
        X, y_pwave, test_size=0.3, random_state=42  # 30% for val+test
    )
    X_val, X_test, y_pwave_val, y_pwave_test = train_test_split(
        X_temp, y_pwave_temp, test_size=0.5, random_state=42  # 15% each
    )
    
    # Split for S-wave regression (70/15/15)
    X_train_s, X_temp_s, y_swave_train, y_swave_temp = train_test_split(
        X, y_swave, test_size=0.3, random_state=42  # 30% for val+test
    )
    X_val_s, X_test_s, y_swave_val, y_swave_test = train_test_split(
        X_temp_s, y_swave_temp, test_size=0.5, random_state=42  # 15% each
    )
    
    logging.info(f'Data splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')
    
    # Save test data for evaluation USING PICKLE
    test_data = {
        'X_test_pwave': X_test,
        'y_test_pwave': y_pwave_test,
        'X_test_swave': X_test_s, 
        'y_test_swave': y_swave_test
    }
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(paths['test_data']), exist_ok=True)
    
    # Save using pickle
    with open(paths['test_data'], 'wb') as f:  # 'wb' = write binary
        pickle.dump(test_data, f)
    
    logging.info(f'Test data saved to: {paths["test_data"]}')
    # Train P-wave classifier
    print("="*50)
    print("TRAINING P-WAVE CLASSIFIER")
    print("="*50)
    
    with mlflow.start_run(run_name="pwave_classifier_comparison"):
        classifier = PWaveClassifier()
        classifier.build_models()
        results = classifier.train_all(X_train, y_pwave_train, X_val, y_pwave_val)
        best_name, best_model = classifier.select_best_model()
        classifier.save_best_model()
        
        # Log best model info
        logging.info('models param and metric logged in mlflow for p wave ')
        mlflow.log_param("best_pwave_model", best_name)
        mlflow.log_metric("best_pwave_roc_auc", results[best_name]['metrics']['roc_auc'])
    
    # Train S-wave regressor
    print("\n" + "="*50)
    print("TRAINING S-WAVE REGRESSOR")
    print("="*50)
    
    with mlflow.start_run(run_name="swave_regressor_comparison"):
        regressor = SWaveRegressor()
        regressor.build_models()
        results = regressor.train_all(X_train_s, y_swave_train, X_val_s, y_swave_val)
        best_name, best_model = regressor.select_best_model(metric='rmse')
        regressor.save_best_model()
        
        # Log best model info
        logging.info('models param and metric logged in mlflow for s wave')
        mlflow.log_param("best_swave_model", best_name)
        mlflow.log_metric("best_swave_rmse", results[best_name]['metrics']['rmse'])
    
    print("\n Training complete! Check MLflow for experiment details.")

if __name__ == "__main__":
    train_earthquake_models()