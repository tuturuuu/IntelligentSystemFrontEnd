import streamlit as st
import pandas as pd
from datetime import date
from plotly import graph_objs as go
import os

st.title('Stock Forecast App')

# Define stock datasets (historical data)
stock_files = {
    "Apple": "data/Apple_20120101_to_20241212.csv",
    "Amazon": "data/Amazon_20120101_to_20241212.csv",
    "Microsoft": "data/Microsoft_20120101_to_20241212.csv",
    "Google": "data/Google_20120101_to_20241212.csv"
}

# Define forecast datasets
forecast_files = {
    "Apple": "predict_data/apple_predictions.csv",
    "Amazon": "predict_data/amazon_predictions.csv",
    "Microsoft": "predict_data/microsoft_predictions.csv",
    "Google": "predict_data/google_predictions.csv"
}

# Define metrics for each company
metrics = {
    "Apple": {"MAE": 2.23, "RMSE": 3.07, "MAPE": 1.30},
    "Amazon": {"MAE": 3.15, "RMSE": 4.52, "MAPE": 1.75},
    "Microsoft": {"MAE": 1.98, "RMSE": 2.89, "MAPE": 1.12},
    "Google": {"MAE": 2.76, "RMSE": 3.85, "MAPE": 1.50}
}

# Dropdown for stock selection (same for historical & forecast)
selected_stock = st.selectbox("Select a stock dataset", list(stock_files.keys()))

# Load historical data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

st.text('Loading stock data...')
data = load_data(stock_files[selected_stock])
st.text('Loading stock data... done!')

if data is not None:
    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        if {'date', 'open', 'close'}.issubset(data.columns):
            fig.add_trace(go.Scatter(x=data['date'], y=data['open'], name="Stock Open"))
            fig.add_trace(go.Scatter(x=data['date'], y=data['close'], name="Stock Close"))
            fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
        else:
            st.error("CSV file must contain 'date', 'open', and 'close' columns.")

    plot_raw_data()
else:
    st.info("Please select a valid dataset.")

# Load Forecast Data
if selected_stock in forecast_files:
    forecast_file = forecast_files[selected_stock]

    def load_forecast(file_path):
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            return data
        else:
            st.warning(f"Forecast file not found: {file_path}")
            return None

    st.text('Loading forecast data...')
    forecast = load_forecast(forecast_file)
    st.text('Loading forecast data... done!')

    if forecast is not None:
        st.subheader('Forecast Data')
        st.write(forecast.tail())

        def plot_forecast():
            fig = go.Figure()

            if 'date' not in forecast.columns or 'predicted' not in forecast.columns or 'actual' not in forecast.columns:
                st.error("Missing required columns in forecast data. Check your CSV file.")
                return

            # Plot Actual Prices (black line, slightly thinner)
            fig.add_trace(go.Scatter(
                x=forecast['date'], 
                y=forecast['actual'], 
                name="Actual Price", 
                mode='lines', 
                line=dict(color='black', width=1)  # Thin black line
            ))

            # Plot Predicted Prices (red line, slightly thicker)
            fig.add_trace(go.Scatter(
                x=forecast['date'], 
                y=forecast['predicted'], 
                name="Predicted Price", 
                mode='lines', 
                line=dict(color='red', width=2)  # Thicker red line
            ))

            # Plot confidence intervals (light blue shaded area)
            if 'upper_bound' in forecast.columns and 'lower_bound' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast['date'], 
                    y=forecast['upper_bound'], 
                    name="Upper Bound", 
                    line=dict(dash='dot', color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['date'], 
                    y=forecast['lower_bound'], 
                    name="Lower Bound", 
                    line=dict(dash='dot', color='blue')
                ))

            # Improve layout
            fig.layout.update(
                title_text=f'Forecast for {selected_stock}', 
                xaxis_rangeslider_visible=True, 
                xaxis_title="Date", 
                yaxis_title="Price",
                legend=dict(x=0, y=1)  # Move legend to top-left
            )

            st.plotly_chart(fig)

        plot_forecast()

    else:
        st.info("No forecast data available for the selected stock.")


# Get selected stock's metrics
selected_metrics = metrics.get(selected_stock, {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"})

# Display metrics in Streamlit
st.subheader(f"{selected_stock} Model Performance Metrics")
st.write(f"**MAE:** {selected_metrics['MAE']}")
st.write(f"**RMSE:** {selected_metrics['RMSE']}")
st.write(f"**MAPE:** {selected_metrics['MAPE']}%")