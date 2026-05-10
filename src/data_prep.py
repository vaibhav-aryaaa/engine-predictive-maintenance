import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, FEATURE_COLUMN, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
)

def prepare_data():
    print(f"Reading data from {RAW_DATA_PATH}...")
    df= pd.read_csv(RAW_DATA_PATH)

    initial_count= len(df)
    df= df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows.")

    X= df[FEATURE_COLUMN]
    y= df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=TEST_SIZE,random_state=RANDOM_STATE, stratify=y
    )

    train_df= pd.concat([X_train,y_train], axis=1)
    test_df=pd.concat([X_test,y_test], axis=1)

    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"Saved processed data to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    prepare_data()