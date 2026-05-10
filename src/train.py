import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import(
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FEATURE_COLUMN,
    TARGET_COLUMN,
    MLFLOW_EXPERIMENT_NAME
)

def train_model():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df=pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df=pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    X_train, y_train = train_df[FEATURE_COLUMN], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMN], test_df[TARGET_COLUMN]

    with mlflow.start_run():
        n_estimators= 200
        max_depth=10

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model= RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)
        acc= accuracy_score(y_test,y_pred)

        mlflow.log_metric("accuracy", acc)
        print(f"Model Accuracy: {acc:.4f}")

        model_path= MODELS_DIR / "engine_model.joblib"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "model")
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()