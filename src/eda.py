import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import RAW_DATA_PATH, FEATURE_COLUMN, TARGET_COLUMN

def run_eda():
    df=pd.read_csv(RAW_DATA_PATH)
    print(df.info())

    print("\nTarget Distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True))