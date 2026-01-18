import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
from lightgbm import LGBMRegressor
import joblib
from typing import Dict
from logger import logging

class SWaveRegressor:
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def get_model_configs(self) -> Dict:
        return {
            'xgboost': {
                'model_class': xgb.XGBRegressor,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'random_forest': {
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'model_class': LGBMRegressor,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'ridge_regression': {
                'model_class': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
    
    def build_models(self):
        configs = self.get_model_configs()
        for name, config in configs.items():
            model_class = config['model_class']
            params = config['params']
            self.models[name] = model_class(**params)
        print(f" Built {len(self.models)} regression models: {list(self.models.keys())}")
    
    def train_all(self, X_train, y_train, X_val, y_val) -> Dict:
        self.results = {}
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate regression metrics
            metrics = {
                'mse': mean_squared_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }
            
            # Store results
            self.results[name] = {
                'model': model,
                'metrics': metrics
            }
            
            print(f"   RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return self.results
    
    def select_best_model(self, metric='r2'):
        if not self.results:
            raise ValueError("Train models first using train_all()")
        
        # For MSE/RMSE/MAE, lower is better
        # For R², higher is better
        if metric in ['mse', 'rmse', 'mae']:
            best_score = float('inf')
            compare = lambda x, y: x < y
        else:  # r2
            best_score = -float('inf')
            compare = lambda x, y: x > y
        
        best_model_name = None
        
        for name, result in self.results.items():
            score = result['metrics'][metric]
            if compare(score, best_score):
                best_score = score
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n Best Regression Model: {best_model_name}")
        print(f"   {metric}: {best_score:.4f}")
        logging.info('---3--- :  model trainner for s wave ')
        logging.info(f'best model name : {best_model_name}')
        
        return best_model_name, self.best_model
    
    def save_best_model(self, path="models/best_swave_model.pkl"):
        if self.best_model is None:
            self.select_best_model()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        logging.info(f'best  model saved at : {path}')
        print(f" Best regression model saved to {path}")