import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV # <--- The Super-Tune
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import numpy as np
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, 
    FEATURE_COLUMN, TARGET_COLUMN, 
    MLFLOW_EXPERIMENT_NAME
)

def train_model():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Load and Scale
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_train, y_train = train_df[FEATURE_COLUMN], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMN], test_df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run():
        # 2. Define the Search Grid
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [6, 8],
            'learning_rate': [0.05, 0.01],
            'scale_pos_weight': [0.6, 1]
        }
        
        # 3. Run the Search
        print("Starting Systematic Tuning (GridSearch)... this may take a minute.")
        grid_search = GridSearchCV(
            XGBClassifier(random_state=42), 
            param_grid, cv=3, scoring='accuracy'
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # 4. Use the Best Model
        model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        
        # 5. Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nBest Params Found: {grid_search.best_params_}")
        print(f"New Model Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)
        print(classification_report(y_test, y_pred))

        # 6. Feature Importance (NOW we calculate it!)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [FEATURE_COLUMN[i] for i in indices]

        print("\n--- Feature Importance ---")
        for f, imp in zip(sorted_features, importances[indices]):
            print(f"{f}: {imp:.4f}")
        
        # Plot and save
        plt.figure(figsize=(10,6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), sorted_features, rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # 7. Save model
        joblib.dump(model, MODELS_DIR / "engine_model.joblib")
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        mlflow.sklearn.log_model(model, "model")
        
if __name__ == "__main__":
    train_model()
