import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"
RAW_DATA_PATH = DATA_DIR / "nasa_raw" / "train_FD001.csv"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
MODELS_DIR=PROJECT_ROOT/"models"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN="Engine Condition"
FEATURE_COLUMN = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21', 'Temp_Diff'
]
NASA_COLUMNS = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]

RANDOM_STATE= 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT_NAME="engine_maintenance_v1"