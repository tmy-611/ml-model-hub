import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import numpy as np

st.title("ML Model Hub")

MODEL_OPTIONS = {
    "Linear Regression": "linear_regression",
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest"
}

# Map metric keys to display names
METRIC_DISPLAY_NAMES = {
    "mean_squared_error": "Mean Squared Error",
    "root_mean_squared_error": "Root Mean Squared Error",
    "r2_score": "R² Score",
    "mean_absolute_error": "Mean Absolute Error",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score"
}

# Custom CSS for table styling
st.markdown(
    """
    <style>
    .stDataFrame {
        font-size: 14px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stDataFrame thead th {
        background-color: #f4f4f4;
        font-weight: bold;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .stButton > button {
        display: block;
        margin: 0 auto;
        background-color: #36A2EB;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Display dataset preview if a file is uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
        # Reset file pointer to the beginning for backend processing
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.error("Please upload a valid CSV file.")

# Target column input
target_column = st.text_input("Enter Target Column Name")

# Model selection
model_select = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))

# Clarify default metrics setting
st.info("Unconfigured models default to regression metrics (MSE, RMSE, R², MAE).")

# Help section for metrics explanation
with st.expander("Learn About Model Metrics"):
    st.markdown("""
        Each model uses specific metrics:
        - **Linear Regression**: Regression metrics (Mean Squared Error, Root Mean Squared Error, R² Score, Mean Absolute Error).
        - **Logistic Regression, Random Forest**: Classification metrics (Accuracy, Precision, Recall, F1 Score).
        If a new model is added without a specified metric type, it will default to regression metrics.
    """)


if st.button("Train Model"):
    if uploaded_file and target_column:
        # Prepare file and parameters
        model_type = MODEL_OPTIONS[model_select]
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        params = {"model_type": model_type, "target_column": target_column}

        # Send request to FastAPI
        try:
            response = requests.post("http://localhost:8000/train", files=files, params=params)
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                st.caption("Metrics for regression: MSE, RMSE, R², MAE. For classification: Accuracy, Precision, Recall, F1.")
                metrics = result["metrics"]
                # Format metrics as a DataFrame with display names
                metrics_df = pd.DataFrame(
                    [(METRIC_DISPLAY_NAMES.get(k, k), v) for k, v in metrics.items()],
                    columns=["Metric", "Value"]
                ).round(4)  # Round to 4 decimal places
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                # Display confusion matrix for classification models
                if "confusion_matrix" in result and result["confusion_matrix"]:
                    st.subheader("Confusion Matrix")
                    cm = np.array(result["confusion_matrix"])
                    # Get unique labels from the dataset for axis labels
                    df = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)  # Reset pointer again
                    labels = sorted(df[target_column].unique())
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=labels,
                        y=labels,
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted Label",
                        yaxis_title="Actual Label"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display and download results
                if os.path.exists(result["results_path"]):
                    results_df = pd.read_csv(result["results_path"])
                    st.subheader("Prediction Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        label="Download Predictions",
                        data=open(result["results_path"], "rb"),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                if os.path.exists(result["model_path"]):
                    st.download_button(
                        label="Download Model",
                        data=open(result["model_path"], "rb"),
                        file_name="model.pkl",
                        mime="application/octet-stream"
                    )
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload a file and specify a target column")