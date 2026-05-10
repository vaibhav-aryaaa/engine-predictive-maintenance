import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"
RAW_DATA_PATH=DATA_DIR/"engine_data.csv"
PROCESSED_DATA_DIR=DATA_DIR/"processed"
MODELS_DIR=PROJECT_ROOT/"models"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN="Engine Condition"
FEATURE_COLUMN=[
    "Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"
]

RANDOM_STATE= 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT_NAME="engine_maintenance_v1"