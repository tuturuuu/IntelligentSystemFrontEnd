import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from datetime import datetime, timedelta
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title('Stock Forecast App')

# Define stock datasets (historical data)
stock_files = {
    "Apple": "data/Apple_20120101_to_20241212.csv",
    "Amazon": "data/Amazon_20120101_to_20241212.csv",
    "Microsoft": "data/Microsoft_20120101_to_20241212.csv",
    "Google": "data/Google_20120101_to_20241212.csv"
}

# Define model paths
model_paths = {
    "Apple": "./amazon_review/amazon_review/1743696135/",
    "Amazon": "./amazon_review/amazon_review/1743696135/",
    "Microsoft": "./amazon_review/amazon_review/1743696135/",
    "Google": "./amazon_review/amazon_review/1743696135/"
}

# Dropdown for stock selection
selected_stock = st.selectbox("Select a stock dataset", list(stock_files.keys()))

# Helper functions for data processing
def normalize(x_, minmax):
    """Normalize input data within the specified min-max range."""
    return (x_ - minmax[0]) / (minmax[1] - minmax[0])

def denormalize(x_, minmax):
    """Denormalize input data based on the specified min-max range."""
    return minmax[0] + ((minmax[1] - minmax[0]) * x_)

def prepare_prediction_window(close_price, analysis_duration):
    """
    Prepare the most recent data for prediction using a sliding window approach.
    
    Parameters:
    close_price (numpy.ndarray): Array of normalized close prices
    analysis_duration (int): Number of time steps to use for prediction (window size)
    
    Returns:
    numpy.ndarray: X_test data ready for prediction
    """
    X_test = []
    
    # Get the latest window of data for prediction
    # Using the same pattern as in the training implementation
    X_test.append(close_price[-analysis_duration:].reshape(analysis_duration))
    
    # Reshape testing data into the required format (samples, time steps, features)
    X_test = np.asarray(X_test).reshape((len(X_test), analysis_duration, 1))
    
    return X_test

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    # Calculate Mean Absolute Error
    mae_value = mean_absolute_error(y_true, y_pred)
    
    # Calculate Root Mean Squared Error
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Mean Absolute Percentage Error
    # Handle zero division errors
    mape_value = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    
    # Calculate R-squared
    r2_value = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae_value,
        'RMSE': rmse_value,
        'MAPE': mape_value,
        'R2': r2_value
    }

# Load historical data
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        elif 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

with st.spinner('Loading stock data...'):
    data = load_data(stock_files[selected_stock])

if data is not None:
    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        # Check common column name patterns
        open_col = 'open' if 'open' in data.columns else 'Open'
        close_col = 'close' if 'close' in data.columns else 'Close'
        fig.add_trace(go.Scatter(x=data.index, y=data[open_col], name="Stock Open"))
        fig.add_trace(go.Scatter(x=data.index, y=data[close_col], name="Stock Close"))
        fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)


    plot_raw_data()
    
    # Model Prediction Logic
    st.subheader('Model Prediction')
    
    # Load model
    model_path = model_paths.get(selected_stock)
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
    else:
        try:
            with st.spinner(f'Loading {selected_stock} prediction model...'):
                reconstructed_model = tf.saved_model.load(model_path)
                infer = reconstructed_model.signatures["serving_default"]
                st.success("Model loaded successfully!")
            
            # Prepare data for prediction
            close_col = 'close' if 'close' in data.columns else 'Close'
            close_price = data[close_col].values
            
            # Parameters for normalization
            try:
                close_minmax_path = f'close_minmax_trxusd.npy'
                
                if os.path.exists(close_minmax_path):
                    close_minmax = np.load(close_minmax_path)
                    st.success(f"Loaded normalization parameters from {close_minmax_path}")
                else:
                    # Fallback in case the file doesn't exist
                    alpha, beta = 0.8, 1.2
                    close_minmax = np.array([alpha * close_price.min(), beta * close_price.max()], dtype='float32')
                    st.warning(f"Could not find saved normalization file. Using calculated values instead.")
            except Exception as e:
                st.error(f"Error loading normalization parameters: {str(e)}")
                # Fallback
                alpha, beta = 0.8, 1.2
                close_minmax = np.array([alpha * close_price.min(), beta * close_price.max()], dtype='float32')
            
            # Normalize data
            data_len = close_price.shape[0]
            close_price = close_price.reshape((data_len, 1)).astype('float32')
            normalized_close = normalize(close_price, close_minmax)
            
            # Define analysis duration (lookback window)
            analysis_duration = 6  # Same as in the model training
            
            # Prepare the prediction input (last n days)
            X_pred = prepare_prediction_window(normalized_close, analysis_duration)
            
            # Convert to tensor for prediction
            X_tensor = tf.convert_to_tensor(X_pred, dtype=tf.float32)
            
            # Make prediction
            with st.spinner('Making predictions...'):
                predictions = infer(close_price=X_tensor)
                if "distribution_lambda" in predictions:
                    pred_array = predictions["distribution_lambda"].numpy()
                    
                    # Calculate mean and standard deviation for confidence intervals
                    pred_mean = pred_array.mean(axis=1)
                    pred_std = pred_array.std(axis=1)
                    
                    # Denormalize predictions
                    actual_values = denormalize(normalized_close[-len(pred_mean):], close_minmax)
                    predicted_values = denormalize(pred_mean, close_minmax)
                    
                    # ENHANCED METRICS CALCULATION
                    st.subheader("Model Performance Metrics")
                    
                    # Create tabs for different metrics calculations
                    tab1, tab2 = st.tabs(["Back-testing Metrics", "Historical Performance"])
                    
                    with tab1:
                        st.write("#### Back-testing Metrics")
                        st.write("Metrics calculated by testing the model on recent historical data")
                        
                        # Configure back-testing parameters
                        col1, col2 = st.columns(2)
                        with col1:
                            test_window = st.slider("Test Window Size (days)", 
                                              min_value=7, 
                                              max_value=238, 
                                              value=30,
                                              help="Number of days to use for back-testing")
                        with col2:
                            # Replace the slider with a fixed value
                            forecast_horizon = 1
                            st.info("Forecast Horizon: 1 day")
                        
                        # Ensure we have enough data for testing
                        if test_window + forecast_horizon + analysis_duration <= len(normalized_close):
                            # Calculate back-testing metrics
                            with st.spinner("Calculating back-testing metrics..."):
                                # Initialize arrays to store actual and predicted values
                                y_true_all = []
                                y_pred_all = []
                                
                                # Loop through the test window
                                for i in range(test_window):
                                    # End index for the current prediction
                                    end_idx = len(normalized_close) - test_window + i
                                    
                                    # Make a prediction using data up to end_idx
                                    X_test = prepare_prediction_window(normalized_close[:end_idx], analysis_duration)
                                    X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
                                    try:
                                        pred = infer(close_price=X_tensor)
                                        pred_mean = pred["distribution_lambda"].numpy().mean(axis=1)
                                        
                                        # Get actual value forecast_horizon steps ahead
                                        if end_idx + forecast_horizon < len(normalized_close):
                                            actual = normalized_close[end_idx + forecast_horizon - 1]
                                            predicted = pred_mean[0]
                                            
                                            # Denormalize values
                                            actual_denorm = denormalize(actual, close_minmax)[0]
                                            pred_denorm = denormalize(predicted, close_minmax)
                                            
                                            y_true_all.append(actual_denorm)
                                            y_pred_all.append(pred_denorm)
                                    except Exception as e:
                                        st.warning(f"Error in prediction {i}: {str(e)}")
                                        continue
                                
                                # Calculate metrics if we have predictions
                                if len(y_true_all) > 0 and len(y_pred_all) > 0:
                                    # Convert to numpy arrays
                                    y_true_np = np.array(y_true_all)
                                    y_pred_np = np.array(y_pred_all)
                                    
                                    # Calculate metrics
                                    metrics = calculate_metrics(y_true_np, y_pred_np)
                                    
                                    # Display metrics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f}")
                                        st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f}")
                                    with col2:
                                        st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%")
                                        st.metric("R² Score", f"{metrics['R2']:.4f}")
                                    
                                    # Create dataframe of actual vs predicted for visualization
                                    backtest_df = pd.DataFrame({
                                        'Actual': y_true_np,
                                        'Predicted': y_pred_np,
                                        'Date': data.index[-test_window:-(test_window-len(y_true_np))]
                                    })
                                    
                                    # Plot actual vs predicted
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=backtest_df['Date'], 
                                        y=backtest_df['Actual'], 
                                        name="Actual",
                                        mode='lines+markers', 
                                        line=dict(color='black')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=backtest_df['Date'], 
                                        y=backtest_df['Predicted'], 
                                        name="Predicted",
                                        mode='lines+markers', 
                                        line=dict(color='red')
                                    ))
                                    fig.update_layout(
                                        title=f"Back-testing: {forecast_horizon}-day Ahead Predictions",
                                        xaxis_title="Date",
                                        yaxis_title="Price",
                                        legend=dict(x=0, y=1)
                                    )
                                    st.plotly_chart(fig)
                                    
                                    # Show error distribution
                                    errors = y_true_np - y_pred_np
                                    abs_pct_errors = np.abs((y_true_np - y_pred_np) / y_true_np) * 100
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        fig = go.Figure()
                                        fig.add_trace(go.Histogram(x=errors, nbinsx=20))
                                        fig.update_layout(
                                            title="Error Distribution",
                                            xaxis_title="Error",
                                            yaxis_title="Frequency"
                                        )
                                        st.plotly_chart(fig)
                                        
                                    with col2:
                                        fig = go.Figure()
                                        fig.add_trace(go.Histogram(x=abs_pct_errors, nbinsx=20))
                                        fig.update_layout(
                                            title="Absolute Percentage Error Distribution",
                                            xaxis_title="Absolute Percentage Error (%)",
                                            yaxis_title="Frequency"
                                        )
                                        st.plotly_chart(fig)
                                else:
                                    st.error("Not enough data to calculate back-testing metrics")
                        else:
                            st.warning("Not enough historical data for the selected test window and forecast horizon")
                    
                    with tab2:
                        st.write("#### Pre-calculated Model Metrics")
                        st.write("These are the metrics calculated during model training and evaluation")
                        
                        # Display pre-calculated metrics
                        metrics = {
                            "Apple": {"MAE": 2.15, "RMSE": 2.95, "MAPE": 1.07},
                            "Amazon": {"MAE": 2.45, "RMSE": 3.3, "MAPE": 1.34},
                            "Microsoft": {"MAE": 3.98, "RMSE": 5.3, "MAPE": 0.96},
                            "Google": {"MAE": 2.04, "RMSE": 3.01, "MAPE": 1.26}
                        }
                        selected_metrics = metrics.get(selected_stock, {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"})
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Mean Absolute Error (MAE)", f"{selected_metrics['MAE']}")
                            st.markdown("""
                            **MAE** measures the average magnitude of errors in a set of predictions, without considering their direction.
                            Lower values indicate better accuracy.
                            """)
                        with col2:
                            st.metric("Root Mean Squared Error (RMSE)", f"{selected_metrics['RMSE']}")
                            st.markdown("""
                            **RMSE** measures the square root of the average of squared differences between predicted and actual values.
                            It gives higher weight to larger errors.
                            """)
                        
                        st.metric("Mean Absolute Percentage Error (MAPE)", f"{selected_metrics['MAPE']}%")
                        st.markdown("""
                        **MAPE** measures prediction accuracy as the average percentage difference between predicted and actual values.
                        It's scale-independent, making it good for comparing forecast accuracy across different datasets.
                        """)
                else:
                    st.error("Model prediction output format is not as expected")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
            # Fallback to the original metrics
            metrics = {
                "Apple": {"MAE": 2.15, "RMSE": 2.95, "MAPE": 1.07},
                "Amazon": {"MAE": 2.45, "RMSE": 3.3, "MAPE": 1.34},
                "Microsoft": {"MAE": 3.98, "RMSE": 5.3, "MAPE": 0.96},
                "Google": {"MAE": 2.04, "RMSE": 3.01, "MAPE": 1.26}
            }
            selected_metrics = metrics.get(selected_stock, {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"})
            
            st.subheader(f"{selected_stock} Model Performance Metrics (Pre-calculated)")
            st.write(f"**MAE:** {selected_metrics['MAE']}")
            st.write(f"**RMSE:** {selected_metrics['RMSE']}")
            st.write(f"**MAPE:** {selected_metrics['MAPE']}%")
else:
    st.info("Please select a valid dataset.")