import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, PROCESSED_DATA_DIR, FEATURE_COLUMN, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

def prepare_nasa_data():
    print(f"Loading NASA raw data from {RAW_DATA_PATH}...")
    
    # 1. Load the CSV
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 2. Clean and Standardize Column Names
    # Remove the first index column if it exists
    if "Unnamed: 0" in df.columns or df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
        
    # Create a mapping to fix spaces (e.g., 'unit number' -> 'unit_number')
    new_names = {
        'unit number': 'unit_number',
        'time in cycles': 'time_in_cycles',
    }
    # Add sensors: 'sensor measurement 1' -> 'sensor_1'
    for i in range(1, 22):
        new_names[f'sensor measurement {i}'] = f'sensor_{i}'
        
    df = df.rename(columns=new_names)
    
    print("Column names standardized.")
    
    # 3. CALCULATE RUL (The "Prognostics" logic)
    # Find the maximum cycle reached by each engine (unit_number)
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    
    # 4. Create Binary Target
    # We define 'Machine failure' (1) if the engine is within 30 cycles of its end of life
    df[TARGET_COLUMN] = (df['RUL'] <= 30).astype(int)
    
    # 5. Feature Engineering
    # Adding a custom interaction feature (Oil Temp vs Coolant Temp equivalent)
    df['Temp_Diff'] = df['sensor_12'] - df['sensor_14']
    
    # 6. Final Feature Selection
    # Combine the sensors from config with our new custom feature
    final_features = FEATURE_COLUMN + ['Temp_Diff']
    
    X = df[final_features]
    y = df[TARGET_COLUMN]
    
    # 7. Split and Save
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    print(f"✅ NASA Data Preparation Complete!")
    print(f"Total rows processed: {len(df)}")
    print(f"Train/Test split saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    prepare_nasa_data()