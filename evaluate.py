import pickle
import yaml
import numpy as np
import sys
import joblib
from sklearn.metrics import roc_auc_score, mean_squared_error
from logger import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def evaluate_models():
    
    # Load config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config['paths']
    
    print("\n" + "="*60)
    print("MODEL EVALUATION FOR CI/CD")
    print("="*60)
    
    #  load test data
    try:
        with open(paths['test_data'], 'rb') as f:
            test_data = pickle.load(f)
        
        X_test_pwave = test_data['X_test_pwave']
        y_test_pwave = test_data['y_test_pwave']
        X_test_swave = test_data['X_test_swave']
        y_test_swave = test_data['y_test_swave']
        
    except Exception as e:
        print(f" Failed to load test data: {e}")
        return 1
    
    #  EVALUATE P-WAVE CLASSIFIER 
    print("\n Evaluating P-wave Classifier...")
    try:
        # Load model
        pwave_model = joblib.load(paths['model_pwave'])
        
        # Predict
        y_pred_proba = pwave_model.predict_proba(X_test_pwave)[:, 1]
        
        # Calculate ROC-AUC (Best metric for binary classification)
        pwave_roc_auc = roc_auc_score(y_test_pwave, y_pred_proba)
        
        print(f"   ROC-AUC Score: {pwave_roc_auc:.4f}")
        print(f"   Threshold: >0.80 {'model passed' if pwave_roc_auc > 0.80 else 'model failed'}")
        
        if pwave_roc_auc <= 0.80:
            print(f"   FAIL: ROC-AUC {pwave_roc_auc:.4f} <= 0.80")
            return 1
            
    except Exception as e:
        print(f"P-wave evaluation failed: {e}")
        return 1
    
    # EVALUATE S-WAVE REGRESSOR
    print("\n Evaluating S-wave Regressor...")
    try:
        # Load model
        swave_model = joblib.load(paths['model_swave'])
        
        # Predict
        y_pred = swave_model.predict(X_test_swave)
        
        # Calculate R² Score (Best metric for regression - % variance explained)
        ss_res = np.sum((y_test_swave - y_pred) ** 2)
        ss_tot = np.sum((y_test_swave - np.mean(y_test_swave)) ** 2)
        swave_r2 = 1 - (ss_res / ss_tot)
        
        # Also calculate RMSE for context
        rmse = np.sqrt(mean_squared_error(y_test_swave, y_pred))
        
        print(f"   R² Score: {swave_r2:.4f}")
        print(f"   RMSE: {rmse:.2f} seconds")
        print(f"   Threshold: R2 >0.80 {'model passed' if swave_r2 > 0.80 else 'model failed'}")
        
        if swave_r2 <= 0.80:
            print(f"   FAIL: R2 {swave_r2:.4f} <= 0.80")
            return 1
            
    except Exception as e:
        print(f" S-wave evaluation failed: {e}")
        return 1
    
    # ========== SUCCESS ==========
    print("\n" + "="*60)
    print("ALL MODELS PASSED THRESHOLDS!")
    print("="*60)
    print(f"P-wave Classifier: ROC-AUC = {pwave_roc_auc:.4f} (>0.80)")
    print(f"S-wave Regressor: R² = {swave_r2:.4f} (>0.80), RMSE = {rmse:.2f}s")
    print("\n Ready for deployment!")
    
    return 0

def main():

    logging.info("Starting model evaluation for CI/CD...")
    
    exit_code = evaluate_models()
    
    # Exit with code for Git Actions
    sys.exit(exit_code)

if __name__ == "__main__":
    main()