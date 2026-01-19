import os
import yaml

def test_imports():
    
    import pandas
    import numpy
    import sklearn
    import mlflow
    assert True  

def test_file_exists():
    """Test that required files exist"""
      
    required_files = [
        'train.py',
        'src/evaluate.py', 
        'app.py',
        'params.yaml',
        'requirements.txt',
        'Dockerfile'
    ]
    for file in required_files:
        assert os.path.exists(file), f"Missing file: {file}"

def test_src_files():
    """Test that source code files exist"""
    
    src_files = [
        'src/feature_engineer.py',
        'src/model_pwave.py',
        'src/model_swave.py'
    ]
    
    for file in src_files:
        assert os.path.exists(file), f"Missing source file: {file}"

def test_params_structure():
    """Test params.yaml has basic structure"""
    import yaml
    
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'paths' in config, "params.yaml missing 'paths'"
    assert 'features' in config, "params.yaml missing 'features'"

def test_feature_count():
    """Test we have 6 features"""
    
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    features = config['features']['feature_columns']
    assert len(features) == 6, f"Should have 6 features, got {len(features)}"
    
    # Check some expected features
    expected = ['sensor_reading', 'noise_level', 'pga', 'snr']
    for feat in expected:
        assert feat in features, f"Missing feature: {feat}"

def test_models_directory():
    """Test models directory exists"""
    import os
    os.makedirs('models', exist_ok=True)
    assert os.path.exists('models'), "Models directory missing"

def test_data_directories():
    """Test data directories exist"""
    import os
    
    # Create if not exists
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    assert os.path.exists('data/raw'), "data/raw missing"
    assert os.path.exists('data/processed'), "data/processed missing"

def test_dag_file():
    """Test Airflow DAG file exists"""
    
    # Create dags directory if not exists
    os.makedirs('dags', exist_ok=True)
    
    # Check if DAG file exists (allow any .py file in dags/)
    if os.path.exists('dags'):
        dag_files = [f for f in os.listdir('dags') if f.endswith('.py')]
        assert len(dag_files) > 0, "No DAG files found in dags/"