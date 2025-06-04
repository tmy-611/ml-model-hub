# ML Model Hub

## Overview

ML Model Hub is a user-friendly web application that allows users to upload their CSV datasets, select a machine learning model, specify a target feature, and receive prediction results along with model performance metrics. This project aims to simplify the process of basic model training and evaluation for common machine learning tasks.

The application is built with a FastAPI backend for model training and API services, and a Streamlit frontend for the user interface.

## Features

*   **CSV Upload:** Users can easily upload their datasets in CSV format.
*   **Model Selection:** Choose from a selection of common machine learning models:
    *   Linear Regression (for regression tasks)
    *   Logistic Regression (for classification tasks)
    *   Random Forest Classifier (for classification tasks)
*   **Target Feature Specification:** Users can define which column in their dataset is the target variable for prediction.
*   **Automated Training & Evaluation:** The backend automatically splits the data (80% for train, 20% for test), trains the selected model, and evaluates its performance on a test set.
*   **Performance Metrics:**
    *   **Regression:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score, Mean Absolute Error (MAE).
    *   **Classification:** Accuracy, Precision, Recall, F1 Score.
    *   **Confusion Matrix:** Visualized for classification models to understand true/false positives and negatives.
*   **Prediction Results:** View a sample of actual vs. predicted values.
*   **Downloadable Artifacts:**
    *   Download the trained model (`.pkl` file).
    *   Download the prediction results (`.csv` file).
*   **Dataset Preview:** See a quick preview of the uploaded dataset.

## Tech Stack

*   **Backend:**
    *   Python
    *   FastAPI: For building the robust and fast API.
    *   Scikit-learn: For machine learning models and metrics.
    *   Pandas: For data manipulation.
    *   NumPy: For numerical operations.
    *   Joblib: For saving and loading trained models.
*   **Frontend:**
    *   Streamlit: For creating the interactive web application.
    *   Plotly Express: For visualizing the confusion matrix.
*   **Other:**
    *   Requests: For communication between Streamlit and FastAPI.
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tmy-611/ml-model-hub.git
    cd ml-model-hub
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the FastAPI backend:**
    Open a terminal and run:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The backend API will typically be available at `http://localhost:8000`.

2.  **Start the Streamlit frontend:**
    Open a new terminal (ensure your virtual environment is activated) and run:
    ```bash
    streamlit run app.py
    ```
    The Streamlit application will typically open in your web browser at `http://localhost:8501`.

## How to Use

1.  Navigate to the Streamlit application URL in your web browser.
2.  **Upload a CSV file:**
    *   Ensure all columns in your CSV are numerical. Categorical features should be pre-processed (e.g., one-hot encoded or label encoded) before uploading.
3.  **Enter Target Column Name:** Specify the exact name of the column you want to predict.
4.  **Select Model:** Choose from the available models (Linear Regression, Logistic Regression, Random Forest).
5.  **Click "Train Model".**
6.  The application will display:
    *   A success message.
    *   Model performance metrics (relevant to the model type).
    *   A confusion matrix (for classification models).
    *   A table of actual vs. predicted values.
    *   Download buttons for the trained model and prediction results.

## Future Enhancements

*   **Data Preprocessing Options:**
    *   Automatic handling of categorical features (e.g., encoding).
    *   Missing value imputation.
    *   Feature scaling.
*   **More Models:**
    *   Expand the library with more regression models (e.g., SVR, Random Forest Regressor, Gradient Boosting Regressor).
    *   Add more classification models (e.g., SVM, KNN, Gradient Boosting Classifier).
*   **Hyperparameter Tuning:** Allow users to specify or tune model hyperparameters.
*   **Cross-Validation:** Implement k-fold cross-validation for more robust metric evaluation.
*   **Enhanced Visualizations:**
    *   ROC curves for classification.
    *   Actual vs. Predicted plots and Residual plots for regression.
    *   Feature importance plots.
*   **User Accounts/Sessions:** To save and manage user-specific models and results.
*   **Dockerization:** For easier deployment.

## Contributing
This is a personal project primarily for learning and portfolio purposes. However, I'm open to suggestions and discussions!

*   **Issues & Feature Requests:** If you have ideas, find a bug, or want to suggest a feature, please feel free to open an issue. I'll review them as time permits.
*   **Pull Requests:** While I appreciate the interest, please discuss any significant changes or new features in an issue *before* submitting a pull request. This helps ensure it aligns with the project's direction and my current learning goals.
## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---
