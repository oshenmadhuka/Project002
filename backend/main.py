#backend codes
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import json
from typing import List
import io
import os
import warnings
from prophet import Prophet
from lightgbm import LGBMRegressor
import pickle
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import datetime
import plotly.graph_objects as go
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client.forecast_db

# Model storage locations
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Pydantic models for data validation
class ModelResults(BaseModel):
    metrics: dict
    chart_data: dict
    computation_time: dict

# Connect to MongoDB
@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient(MONGO_URI)
    app.database = app.mongodb_client.forecast_db
    print("Connected to MongoDB!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()
    print("MongoDB connection closed")

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Time Series Forecasting API"}

@app.post("/upload/train-data/")
async def upload_train_data(file: UploadFile = File(...)):
    """
    Upload training dataset
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Save to MongoDB
        dataset_id = db.datasets.insert_one({
            "name": file.filename,
            "type": "training",
            "created_at": datetime.datetime.now(),
            "data": df.to_dict(orient="records")
        }).inserted_id
        
        return {"message": "Training data uploaded successfully", "dataset_id": str(dataset_id)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/upload/predict-data/")
async def upload_predict_data(file: UploadFile = File(...)):
    """
    Upload prediction dataset
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Save to MongoDB
        dataset_id = db.datasets.insert_one({
            "name": file.filename,
            "type": "prediction",
            "created_at": datetime.datetime.now(),
            "data": df.to_dict(orient="records")
        }).inserted_id
        
        return {"message": "Prediction data uploaded successfully", "dataset_id": str(dataset_id)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/train-model/{dataset_id}")
async def train_model(dataset_id: str, background_tasks: BackgroundTasks):
    """
    Train model using the uploaded training dataset
    """
    try:
        # Retrieve dataset from MongoDB
        dataset = db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert back to DataFrame
        data = pd.DataFrame(dataset["data"])
        
        # Start background task for model training
        background_tasks.add_task(run_training_process, data, dataset_id)
        
        return {"message": "Model training started in background", "dataset_id": dataset_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.get("/model-results/{dataset_id}")
async def get_model_results(dataset_id: str):
    """
    Get model training results
    """
    try:
        # Check if results exist
        results = db.model_results.find_one({"dataset_id": dataset_id})
        if not results:
            return {"status": "pending", "message": "Model training in progress or not started"}
        
        # Return results
        return {
            "status": "completed",
            "metrics": results["metrics"],
            "chart_data": results["chart_data"],
            "computation_time": results["computation_time"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@app.post("/predict/{model_id}/{dataset_id}")
async def predict(model_id: str, dataset_id: str, background_tasks: BackgroundTasks):
    """
    Generate predictions using trained model and new data
    """
    try:
        # Retrieve prediction dataset
        dataset = db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert back to DataFrame
        data_predict_process = pd.DataFrame(dataset["data"])
        
        # Start background task for prediction
        background_tasks.add_task(run_prediction_process, data_predict_process, model_id, dataset_id)
        
        return {"message": "Prediction process started in background", "dataset_id": dataset_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.get("/prediction-results/{dataset_id}")
async def get_prediction_results(dataset_id: str):
    """
    Get prediction results
    """
    try:
        # Check if results exist
        results = db.prediction_results.find_one({"dataset_id": dataset_id})
        if not results:
            return {"status": "pending", "message": "Prediction in progress or not started"}
        
        # Return results
        return {
            "status": "completed",
            "chart_data": results["chart_data"],
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving prediction results: {str(e)}")

# Background processing functions
def run_training_process(data, dataset_id):
    """
    Background task to run the full training process
    """
    try:
        # Define column names
        time_line = 'time_line'
        response_name = 'response'
        
        # Convert time_line to datetime
        data[time_line] = pd.to_datetime(data[time_line])
        
        # Extract response data
        data_response = data[response_name]
        
        # Extract predictor data
        data_predictor = data.drop([time_line, response_name], axis=1).sort_index(axis=1)
        
        # Create final dataframe
        df = pd.concat([data[time_line], data_response, data_predictor], axis=1)
        df[response_name] = df[response_name].round().astype(int)
        
        # Sort by time
        df = df.sort_values(by='time_line', ascending=True).reset_index(drop=True)
        
        # Drop columns with only one unique value except 'time_line' and 'response'
        df = df.loc[:, df.nunique() > 1]
        
        # Ensure 'time_line' and 'response' columns are retained
        columns_to_keep = ['time_line', 'response']
        df = df[columns_to_keep + [col for col in df.columns if col not in columns_to_keep]]
        
        # Add time features
        df = add_time_features(df, time_line)
        
        # Prophet Model Development
        df_prop = pd.concat([df[time_line], df[response_name]], axis=1)
        df_prop.columns = ['ds', 'y']
        df_prop['ds'] = pd.to_datetime(df_prop['ds'])
        
        # Train-test split
        train_df_prop, test_df_prop = train_test_split_time_series(df_prop, 'ds')
        
        # Fit Prophet model
        start_time = time.time()
        model_prop = Prophet()
        model_prop.fit(train_df_prop)
        prophet_train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        future = model_prop.make_future_dataframe(periods=len(test_df_prop), freq='D')
        forecast_prop = model_prop.predict(future)
        prophet_pred_time = time.time() - start_time
        
        # Merge predictions with original data
        df_prop = df_prop.merge(forecast_prop[['ds', 'yhat']], on='ds', how='left')
        df_prop.rename(columns={'ds': time_line, 'y': response_name, 'yhat': 'fitted_prop'}, inplace=True)
        
        # Compute residuals
        df_prop['residual'] = df_prop['response'] - df_prop['fitted_prop']
        
        # Split for testing
        df_prop_train_fit, df_prop_test_fit = train_test_split_time_series(df_prop, time_line)
        
        # Evaluate Prophet model
        results_df_prop = evaluate_model(df_prop_test_fit, actual_col=response_name, predicted_col="fitted_prop")
        
        # LightGBM model fitting
        df_ml_model = df.copy()
        df_ml_model.rename(columns={"response": "residual"}, inplace=True)
        df_ml_model["residual"] = df_prop["residual"].values
        
        # Remove multicollinearity
        df_ml_model = handle_multicollinearity(df_ml_model)
        
        # Train-test split for ML model
        df_ml_train, df_ml_test = train_test_split_time_series(df_ml_model, time_line)
        
        # Check if residuals are white noise
        white_noise_status = check_white_noise_status(df_ml_train)
        
        # Set index for ML training
        df_ml_train.set_index('time_line', inplace=True, drop=True)
        X_ml_train = df_ml_train.drop(['residual'], axis=1)
        y_ml_train = df_ml_train['residual']
        
        # Train LightGBM model
        start_time = time.time()
        best_lgb_model = train_lightgbm(X_ml_train, y_ml_train)
        lgb_train_time = time.time() - start_time
        
        # Prepare test data
        df_ml_test.set_index('time_line', inplace=True, drop=True)
        X_ml_test = df_ml_test.drop(['residual'], axis=1)
        y_ml_test = df_ml_test['residual']
        
        # Make predictions with LightGBM
        start_time = time.time()
        y_ml_test_pred = best_lgb_model.predict(X_ml_test)
        lgb_pred_time = time.time() - start_time
        
        # Evaluate LightGBM on residuals
        df_predictions_ml = pd.DataFrame({
            "Actual Residual": y_ml_test.values,
            "Predicted Residual": y_ml_test_pred
        }, index=X_ml_test.index)
        
        # Compute final forecast
        df_final_forecast = df_prop_test_fit[[time_line, 'response']].copy()
        df_final_forecast['final_forecast'] = np.round(
            df_predictions_ml['Predicted Residual'].values + df_prop_test_fit['fitted_prop'].values
        )
        
        # Replace negative values with 0
        df_final_forecast['final_forecast'] = df_final_forecast['final_forecast'].apply(lambda x: max(x, 0)).astype(int)
        
        # Evaluate final forecast
        result_df_final = evaluate_model(df_final_forecast, 'response', 'final_forecast')
        
        # Create chart data for visualization
        chart_data = create_chart_data(df_final_forecast)
        
        # Computation time summary
        df_computation_time = pd.DataFrame({
            "Model": ["Prophet", "LightGBM"],
            "Training Time (seconds)": [prophet_train_time, lgb_train_time],
            "Prediction Time (seconds)": [prophet_pred_time, lgb_pred_time]
        })
        
        df_computation_time["Total Time (seconds)"] = (
            df_computation_time["Training Time (seconds)"] + df_computation_time["Prediction Time (seconds)"]
        )
        
        # Save models
        save_models(model_prop, best_lgb_model, dataset_id)
        
        # Save results to MongoDB
        results = {
            "dataset_id": dataset_id,
            "metrics": result_df_final.to_dict(orient="records"),
            "chart_data": chart_data,
            "computation_time": df_computation_time.to_dict(orient="records"),
            "created_at": datetime.datetime.now()
        }
        
        db.model_results.insert_one(results)
        
        print(f"Training completed for dataset {dataset_id}")
        
    except Exception as e:
        print(f"Error in training process: {str(e)}")
        # Log the error to MongoDB
        db.errors.insert_one({
            "dataset_id": dataset_id,
            "process": "training",
            "error": str(e),
            "timestamp": datetime.datetime.now()
        })

def run_prediction_process(data_predict_process, model_id, dataset_id):
    """
    Background task to run the prediction process - updated based on notebook code
    """
    try:
        time_line = 'time_line'
        response_name = 'response'
        
        # Load trained models
        model_prop, best_lgb_model = load_models(model_id)
        
        # Get training data to ensure consistency in processing
        training_data = db.datasets.find_one({"_id": ObjectId(model_id)})
        df = pd.DataFrame(training_data["data"])
        
        # Convert time_line to datetime
        data_predict_process[time_line] = pd.to_datetime(data_predict_process[time_line])
        
        # Extract predictor data
        data_predictor_pred = data_predict_process.drop([time_line], axis=1)
        
        # Create dataframe with all data
        df_predict_process = pd.concat([data_predict_process[time_line], data_predictor_pred], axis=1)
        
        # Select only columns from df that are present in df_predict_process
        selected_columns = [col for col in df.columns if col in df_predict_process.columns and col != response_name]
        df_predict_selected = df_predict_process[selected_columns]
        
        # Prepare data for Prophet
        df_prop_predict = pd.concat([df[time_line], df[response_name]], axis=1)
        df_prop_predict.reset_index(inplace=True, drop=True)
        df_prop_predict.columns = ['ds', 'y']
        df_prop_predict['ds'] = pd.to_datetime(df_prop_predict['ds'])
        
        # Create future dataframe for Prophet
        future_pred = model_prop.make_future_dataframe(periods=len(df_predict_selected), freq='D')
        forecast_prop_future_pred = model_prop.predict(future_pred)
        
        # Extract relevant data
        df_predict_prop_future = forecast_prop_future_pred.tail(len(df_predict_selected))[['ds', 'yhat']].copy()
        df_predict_prop_future.rename(columns={'ds': 'time_line'}, inplace=True)
        
        # Prepare data for LightGBM
        df_predict_selected.set_index('time_line', inplace=True)
        X_future_res = df_predict_selected
        
        # Predict residuals
        y_future_res = best_lgb_model.predict(X_future_res)
        
        # Final future forecast
        df_final_future_result = df_predict_prop_future[['time_line']].copy()
        df_final_future_result['Final_future'] = df_predict_prop_future['yhat'] + y_future_res
        df_final_future_result['Final_future'] = df_final_future_result['Final_future'].apply(lambda x: max(0, round(x)))
        
        # Create chart data
        chart_data = create_future_chart_data(df_final_future_result)
        
        # Save results to MongoDB
        results = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "chart_data": chart_data,
            "created_at": datetime.datetime.now()
        }
        
        db.prediction_results.insert_one(results)
        
        print(f"Prediction completed for dataset {dataset_id}")
        
    except Exception as e:
        print(f"Error in prediction process: {str(e)}")
        # Log the error to MongoDB
        db.errors.insert_one({
            "dataset_id": dataset_id,
            "model_id": model_id,
            "process": "prediction",
            "error": str(e),
            "timestamp": datetime.datetime.now()
        })

# Helper functions
def add_time_features(data_frame, time_column):
    """
    Add time-related features to the dataframe
    """
    data_frame[time_column] = pd.to_datetime(data_frame[time_column], errors='coerce')

    data_frame["month"] = data_frame[time_column].dt.month
    data_frame["day"] = data_frame[time_column].dt.day
    data_frame["day_of_week"] = data_frame[time_column].dt.dayofweek 
    data_frame["week_of_year"] = data_frame[time_column].dt.isocalendar().week
    data_frame["quarter"] = data_frame[time_column].dt.quarter
    data_frame["is_weekend"] = (data_frame[time_column].dt.dayofweek >= 5).astype(int)

    return data_frame

def train_test_split_time_series(df, time_column, train_ratio=0.75):
    """
    Split time series data into train and test sets
    """
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    df = df.sort_values(by=time_column).reset_index(drop=True)
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df

def evaluate_model(df_actual_pred, actual_col, predicted_col):
    """
    Evaluate model performance
    """
    actual = df_actual_pred[actual_col].values
    actual_zero_adjusted = df_actual_pred[actual_col].replace(0, 1).values
    predicted = df_actual_pred[predicted_col].values

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))
    
    ad = abs(actual - predicted)
    wmad = np.sum(ad) / np.sum(actual_zero_adjusted)
    avg_accuracy = 1 - wmad
    
    smape = np.mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-6))

    results = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "Accuracy"],
        "Value": [mae, mse, rmse, mape, smape, avg_accuracy]
    })

    return results

def check_white_noise_status(df):
    """
    Check if residuals are white noise
    """
    residuals = df['residual'].values
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    
    if np.all(lb_test['lb_pvalue'] > 0.05):
        return "White Noise (Residuals are Random)"
    else:
        return "Not White Noise (Residuals Have Patterns)"

def handle_multicollinearity(df_ml_model):
    """
    Remove multicollinearity from the data
    """
    # Exclude 'residual' and 'time_line' from correlation analysis
    independent_vars = df_ml_model.drop(columns=['residual', 'time_line'], errors='ignore')
    
    # Compute Correlation Matrix
    corr_matrix = independent_vars.corr()
    
    # Find highly correlated features
    correlation_threshold = 0.80
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify columns to drop
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    df_ml_model.drop(columns=to_drop, inplace=True, errors='ignore')
    
    # Compute VIF to Remove Additional Multicollinear Variables
    X_vif = df_ml_model.drop(columns=['residual', 'time_line'], errors="ignore")
    
    if not X_vif.empty:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values.astype(float), i) for i in range(X_vif.shape[1])]
        
        # Drop variables with high VIF (Above 10)
        high_vif_features = vif_data[vif_data["VIF"] > 10]["Feature"].tolist()
        df_ml_model.drop(columns=high_vif_features, inplace=True, errors='ignore')
    
    return df_ml_model

def train_lightgbm(X_train, y_train):
    """
    Train LightGBM model with hyperparameter tuning
    """
    # Simple parameter configuration for speed
    params = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': -1
    }
    
    # Create and train the model
    model = LGBMRegressor(**params, random_state=0)
    model.fit(X_train, y_train)
    
    return model

def create_chart_data(df_final_forecast):
    """
    Create chart data for visualization
    """
    df_final_forecast['time_line'] = pd.to_datetime(df_final_forecast['time_line'])
    
    chart_data = {
        "x": df_final_forecast['time_line'].dt.strftime('%Y-%m-%d').tolist(),
        "actual": df_final_forecast['response'].tolist(),
        "predicted": df_final_forecast['final_forecast'].tolist()
    }
    
    return chart_data

def create_future_chart_data(df_final_future_result):
    """
    Create chart data for future prediction visualization
    """
    df_final_future_result['time_line'] = pd.to_datetime(df_final_future_result['time_line'])
    
    chart_data = {
        "x": df_final_future_result['time_line'].dt.strftime('%Y-%m-%d').tolist(),
        "predicted": df_final_future_result['Final_future'].tolist()
    }
    
    return chart_data

def save_models(prophet_model, lgbm_model, dataset_id):
    """
    Save trained models to disk
    """
    # Save Prophet model
    with open(f"{MODEL_DIR}/prophet_{dataset_id}.pkl", "wb") as f:
        pickle.dump(prophet_model, f)
    
    # Save LightGBM model
    with open(f"{MODEL_DIR}/lgbm_{dataset_id}.pkl", "wb") as f:
        pickle.dump(lgbm_model, f)

def load_models(dataset_id):
    """
    Load trained models from disk
    """
    # Load Prophet model
    with open(f"{MODEL_DIR}/prophet_{dataset_id}.pkl", "rb") as f:
        prophet_model = pickle.load(f)
    
    # Load LightGBM model
    with open(f"{MODEL_DIR}/lgbm_{dataset_id}.pkl", "rb") as f:
        lgbm_model = pickle.load(f)
    
    return prophet_model, lgbm_model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)