import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from pathlib import Path
import yaml , os
from logger import logging

logging.info('---1--- :  Feature Engineering started ')

class SeismicFeatureEngineer:
    
    def __init__(self, config_path="params.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.feature_params = self.config["features"]
    
    def load_data(self, data_path):
        """Load raw seismic CSV"""
        logging.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logging.info(f"Data shape: {data.shape}")
        return data
    
    def clean_data(self, df):
        
        logging.info("Cleaning the data..,.")
        
        df = df.drop_duplicates()
        df = df.fillna(method='ffill').fillna(method='bfill')

        df = df[
         (df['sensor_reading'].between(-1000, 1000)) &  # Sensor range
         (df['noise_level'] >= 0) & (df['noise_level'] <= 1) &  # 0-1 normalized
         (df['pga'] >= 0) & (df['pga'] <= 1) &  # PGA normalized 0-1
         (df['snr'].between(-50, 50)) &  # SNR realistic range
         (df['ttf_seconds'] >= 0) & (df['ttf_seconds'] <= 300)  # 0-5 min warning
        ]
        
        df = df[
        (df['sensor_reading'] != 999999) &  # error code
        (df['sensor_reading'] != -999999) &
        (~df['sensor_reading'].isna())
        ]
        
        logging.info(f"After cleaning: {df.shape}")
        return df
    
    def prepare_targets(self, df):
        
        # P-wave cl target
        y_pwave = df[self.feature_params['target_pwave']].values
        
        # S-wave regr target ttf 
        y_swave = df[self.feature_params['target_swave']].values
        
        # Features (exclude targets)
        feature_cols = [col for col in df.columns if col not in 
                       [self.feature_params['target_pwave'], self.feature_params['target_swave']]]
        X = df[feature_cols].values
        
        return X, y_pwave, y_swave
    
    def run_pipeline(self, input_path, output_path):

        # Loading
        df = self.load_data(input_path)
        
        # Clean
        df = self.clean_data(df)
        
        # Save processed data
        filedir, filename = os.path.split(output_path)
        
        if filedir != '':
           os.makedirs(filedir, exist_ok=True)
           
        df.to_csv(output_path, index=False)
        
        logging.info(f"Processed data saved to {output_path}")
        
        X, y_pwave, y_swave = self.prepare_targets(df)
        
        return X, y_pwave, y_swave, df.columns.tolist()

if __name__ == "__main__":
    import yaml
    
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    engineer = SeismicFeatureEngineer()
    
    input_path = params["paths"]["data_raw"]
    output_path = params["paths"]["data_processed"]
    
    X, y_pwave, y_swave, feature_names = engineer.run_pipeline(input_path, output_path)
    
    print(f" Feature engineering complete!")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   P-wave positives: {sum(y_pwave)}/{len(y_pwave)}")
    print(f"   S-wave target range: {y_swave.min():.1f}s to {y_swave.max():.1f}s")