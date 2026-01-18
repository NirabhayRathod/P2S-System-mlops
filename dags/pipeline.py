from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import yaml
import os
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.feature_engineer import SeismicFeatureEngineer


with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)  # Fixed: safe_load, not safe_dump_all
paths = config['paths']

with DAG(
    dag_id='earthquake_system',
    start_date=datetime(2024, 1, 19),  # Fixed: start_date, not start_time
    schedule='@once',
    catchup=False,
    default_args={
        'owner': 'mlops',
        'retries': 1
    }
) as dag:
    
    @task
    def load_data():
     
        data_path = paths['data_raw']
        
        if os.path.exists(data_path):
            print(f"Data found at: {data_path}")
            data = pd.read_csv(data_path)
            print(f"   Loaded {len(data)} rows, {len(data.columns)} columns")
            return {'status': 'success', 'data': data.to_dict(), 'rows': len(data)}
        else: 
            print(f"CRITICAL: Data not found at {data_path}")
            print()
            raise FileNotFoundError(f"CRITICAL: Data not found at {data_path}")
    
    @task
    def preprocessing(load_result):
       
        if load_result['status'] != 'success':
            
            raise ValueError(f"Cannot preprocess: Load failed with status {load_result['status']}")
        
        print("Starting feature preprocessing...")
        
        engineer = SeismicFeatureEngineer()
        try:
            X, y_pwave, y_swave, features = engineer.run_pipeline(
                paths['data_raw'],
                paths['data_processed']
            )
            
            result = {
                'status': 'success',
                'samples_processed': X.shape[0],
                'output_path': paths['data_processed']
            }
            
            print(f"  Preprocessing complete!")
            print(f"  Samples: {result['samples_processed']}")

            return result
            
        except Exception as e:
            error_msg = f" Preprocessing failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    @task
    def train_models(preprocess_result):

        if preprocess_result['status'] != 'success':  # Add this check
            raise ValueError("Cannot train: Preprocessing failed")
    
        from train import train_earthquake_models
        train_earthquake_models()
    
        print('Congrats, training is completed!')
        return {'status': 'success', 'message': 'Training completed'}
        
    

    data_result = load_data()
    features_result = preprocessing(data_result)
    models_result = train_models(features_result)