import os , logging 
from pathlib import Path
list_of_files = [
    "src/__init__.py",
    "src/feature_engineer.py",  
    "src/model_pwave.py",    
    "src/model_swave.py",    
    "src/predictor.py",       
    "dags/training_dag.py",
    "dags/data_dag.py", 
    "train.py",
    "Dockerfile",
    "data/raw/seismic_data.csv",    
    "data/processed/seismic_data.csv", 
    "models/pwave_model.pkl",
    "models/swave_model.pkl",
    ".github/workflows/cicd.yml",
    ".gitignore",
    ".dvcignore",
    "requirements.txt",
    "params.yaml", 
]
for filepath in list_of_files:
    file=Path(filepath)
    filedir , filename= os.path.split(file)
    
    if filedir !='':
        os.makedirs(filedir , exist_ok=True)
    if (not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open(filepath , 'w') as f:
            pass
        print(f'{filepath} created')
    else: print(f'{filepath} already exist')
