from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Model configurations
MODELS = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier
}

# Metric configurations
METRIC_CONFIGS = {
    "regression": {
        "metrics": {
            "mean_squared_error": mean_squared_error,
            "root_mean_squared_error": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score,
            "mean_absolute_error": mean_absolute_error
        }
    },
    "classification": {
        "metrics": {
            "accuracy": accuracy_score,
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    }
}

# Map models to metric types
MODEL_METRIC_TYPES = {
    "linear_regression": "regression",
    "logistic_regression": "classification",
    "random_forest": "classification"
}

def compute_metrics(y_true, y_pred, metric_type, unique_labels=None):   
    """Compute metrics for a given metric type."""
    if metric_type not in METRIC_CONFIGS:
        logger.error(f"Invalid metric type: {metric_type}")
        raise ValueError(f"Invalid metric type: {metric_type}")
    
    metrics = {}
    for metric_name, metric_func in METRIC_CONFIGS[metric_type]["metrics"].items():
        try:
            metrics[metric_name] = metric_func(y_true, y_pred)
            logger.info(f"Computed {metric_name}: {metrics[metric_name]}")
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {str(e)}")
            metrics[metric_name] = "N/A"  # Include error in output for debugging
    
    # Compute confusion matrix for classification
    confusion_matrix_data = None
    if metric_type == "classification":
        try:
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            confusion_matrix_data = cm.tolist()  # Convert to list for JSON serialization
            logger.info(f"Computed confusion matrix: {confusion_matrix_data}")
        except Exception as e:
            logger.error(f"Error computing confusion matrix: {str(e)}")
            confusion_matrix_data = None

    logger.info(f"Final metrics: {metrics}")
    return metrics, confusion_matrix_data

@app.post("/train")
async def train_model(file: UploadFile = File(...), model_type: str = None, target_column: str = None):
    try:
        # Validate model type
        if model_type not in MODELS:
            logger.error(f"Invalid model type: {model_type}")
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Read CSV file
        if not file.filename.endswith('.csv'):
            logger.error("File must be a CSV")
            raise HTTPException(status_code=400, detail="File must be a CSV")
        df = pd.read_csv(file.file)

        # Validate target column
        if target_column not in df.columns:
            logger.error(f"Target column not found: {target_column}")
            raise HTTPException(status_code=400, detail="Target column not found in CSV")

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        unique_labels = y.unique() if model_type in ["logistic_regression", "random_forest"] else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if model_type in ["logistic_regression", "random_forest"] else None)

        # Train model
        model = MODELS[model_type]()
        model.fit(X_train, y_train)
        logger.info(f"Model {model_type} trained successfully")

        # Save model
        model_path = f"models/{model_type}_model.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)

        # Evaluate model
        y_pred = model.predict(X_test)
        metric_type = MODEL_METRIC_TYPES.get(model_type, "regression")  # Default to regression if not specified
        metrics, confusion_matrix_data = compute_metrics(y_test, y_pred, metric_type, unique_labels)
        
        # Save predictions
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        results_path = f"results/{model_type}_results.csv"
        os.makedirs("results", exist_ok=True)
        results_df.to_csv(results_path, index=False)

        response_content = {
            "message": "Model trained successfully",
            "metrics": metrics,
            "model_path": model_path,
            "results_path": results_path
        }
        if confusion_matrix_data is not None:
            response_content["confusion_matrix"] = confusion_matrix_data

        logger.info(f"Returning response with metrics: {metrics}")
        return JSONResponse(content=response_content)
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))