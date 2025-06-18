import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
from tensorflow.keras.models import load_model


# Function to calculate metrics
def print_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    accuracy = 100 - mape  # Accuracy as 100 - MAPE
    return mse, mae, mape, accuracy


# Fusion Function: combines predictions from RandomForest, SVR, and LSTM (if available)
def prediction_fusion(stock, days):
    # Download stock data
    df = yf.download(stock, period='1y')  # Changed to '1y' for 1 year of data
    df['Date'] = pd.to_datetime(df.index)
    df['Prediction'] = df[['Close']].shift(-days)

    # Ensure the dataset is not empty after shifting
    df = df.dropna(subset=['Prediction'])

    X = np.array(df[['Close']])
    X = X[:-days]
    y = np.array(df['Prediction'])
    y = y[:-days]

    # If dataset is too small after filtering, raise an error
    if len(X) == 0:
        raise ValueError("Not enough data for the selected period and days to forecast")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train RandomForest
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Train SVR
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_predictions = svr_model.predict(X_test)

    # Load the LSTM model
    lstm_model = load_model('your_lstm_model.h5')  # Replace with your actual LSTM model path
    lstm_predictions = lstm_model.predict(X_test)

    # Fusion of predictions: averaging
    fused_predictions = (rf_predictions + svr_predictions + lstm_predictions.flatten()) / 3

    # Print evaluation metrics for each model and the fused model
    rf_mse, rf_mae, rf_mape, rf_accuracy = print_metrics(y_test, rf_predictions)
    svr_mse, svr_mae, svr_mape, svr_accuracy = print_metrics(y_test, svr_predictions)
    fused_mse, fused_mae, fused_mape, fused_accuracy = print_metrics(y_test, fused_predictions)

    print(f"Random Forest - MSE: {rf_mse:.2f}, MAE: {rf_mae:.2f}, MAPE: {rf_mape:.2f}%, Accuracy: {rf_accuracy:.2f}%")
    print(f"SVR - MSE: {svr_mse:.2f}, MAE: {svr_mae:.2f}, MAPE: {svr_mape:.2f}%, Accuracy: {svr_accuracy:.2f}%")
    print(
        f"Fused Model - MSE: {fused_mse:.2f}, MAE: {fused_mae:.2f}, MAPE: {fused_mape:.2f}%, Accuracy: {fused_accuracy:.2f}%")

    # Prepare the plot for the fused model's forecast
    forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': fused_predictions})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Fused Forecast'))
    fig.update_layout(title=f"{stock} Fused Stock Price Forecast for the Next {days - 1} Days")

    return fig, fused_mse, fused_mae, fused_mape, fused_accuracy


# Example of calling the fusion function
stock = 'AAPL'  # Example stock
days = 10  # Number of forecast days
fig, fused_mse, fused_mae, fused_mape, fused_accuracy = prediction_fusion(stock, days)

# To display the plot (if you're using Plotly)
fig.show()
