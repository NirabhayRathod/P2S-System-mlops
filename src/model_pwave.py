import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lightgbm import LGBMClassifier
import joblib
from typing import Dict
import pandas as pd
from logger import logging

class PWaveClassifier:
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def get_model_configs(self) -> Dict:
        return {
            'xgboost': {
                'model_class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'random_forest': {
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'lightgbm': {
                'model_class': LGBMClassifier,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model_class': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'max_iter': 1000,
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
        print(f"Built {len(self.models)} models: {list(self.models.keys())}")
    
    def train_all(self, X_train, y_train, X_val, y_val) -> Dict:
        self.results = {}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        for name, model in self.models.items():
            print(f"\n Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1_score': f1_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Store results
            self.results[name] = {
                'model': model,
                'metrics': metrics
            }
            
            print(f"   Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return self.results
    
    def select_best_model(self, metric='roc_auc'):
        if not self.results:
            raise ValueError("Train models first using train_all()")
        
        best_score = -1
        best_model_name = None
        
        for name, result in self.results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n Best Model: {best_model_name}")
        print(f"   {metric}: {best_score:.4f}")
        logging.info('---2--- :  model trainner for p wave ')
        logging.info(f'best model name : {best_model_name}')
        
        return best_model_name, self.best_model
    
    def save_best_model(self, path="models/best_pwave_model.pkl"):
        if self.best_model is None:
            self.select_best_model()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.best_model, path)
        logging.info(f'best model saved at : {path}')
        print(f" Best model saved to {path}")
# Example usage in train.py will be:
# classifier = PWaveClassifier()
# classifier.build_models()
# results = classifier.train_all(X_train, y_train, X_val, y_val)
# best_name, best_model = classifier.select_best_model()
# classifier.save_best_model()