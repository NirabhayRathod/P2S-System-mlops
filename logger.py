import logging 
import os 
from datetime import datetime

path=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log" 
log_dir=os.path.join(os.getcwd() , 'logs')           
os.makedirs(log_dir , exist_ok=True)              
log_file_path=os.path.join(log_dir , path)              

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s]- %(name)s- %(levelname)s -%(message)s"
)