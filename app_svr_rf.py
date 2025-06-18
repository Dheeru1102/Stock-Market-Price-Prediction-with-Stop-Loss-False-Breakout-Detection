import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime as dt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dash.exceptions import PreventUpdate
import numpy as np

app = dash.Dash(__name__)

def get_stock_price_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Opening Price'))
    fig.update_layout(title="Stock Price (Close & Open) Over Time")
    return fig

def get_more(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20'))
    fig.update_layout(title="Exponential Moving Average (EMA-20) Over Time")
    return fig


from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.metrics import mean_squared_error, mean_absolute_error


def prediction(stock, days):
    df = yf.download(stock)
    df['Date'] = pd.to_datetime(df.index)
    df['Prediction'] = df[['Close']].shift(-days)
    X = np.array(df[['Close']])
    X = X[:-days]
    y = np.array(df['Prediction'])
    y = y[:-days]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train Random Forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Train Support Vector Regression (SVR) model
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_predictions = svr_model.predict(X_test)

    # Calculate and print evaluation metrics for Random Forest
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mape = np.mean(np.abs((y_test - rf_predictions) / y_test)) * 100
    rf_accuracy = 100 - rf_mape  # Basic "accuracy" for Random Forest

    # Calculate and print evaluation metrics for SVR
    svr_mse = mean_squared_error(y_test, svr_predictions)
    svr_mae = mean_absolute_error(y_test, svr_predictions)
    svr_mape = np.mean(np.abs((y_test - svr_predictions) / y_test)) * 100
    svr_accuracy = 100 - svr_mape  # Basic "accuracy" for SVR

    print(f"Random Forest Metrics:")
    print(f"Mean Squared Error: {rf_mse:.2f}")
    print(f"Mean Absolute Error: {rf_mae:.2f}")
    print(f"Mean Absolute Percentage Error: {rf_mape:.2f}%")
    print(f"Accuracy: {rf_accuracy:.2f}%")
    print("-" * 50)

    print(f"SVR Metrics:")
    print(f"Mean Squared Error: {svr_mse:.2f}")
    print(f"Mean Absolute Error: {svr_mae:.2f}")
    print(f"Mean Absolute Percentage Error: {svr_mape:.2f}%")
    print(f"Accuracy: {svr_accuracy:.2f}%")
    print("-" * 50)

    # Prepare future dates and forecast DataFrame
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    rf_forecast_set = rf_model.predict(np.array(df[['Close']])[-days:])
    svr_forecast_set = svr_model.predict(np.array(df[['Close']])[-days:])

    # Create the figure for plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=rf_forecast_set, mode='lines', name='Random Forest Forecast'))
    fig.add_trace(go.Scatter(x=future_dates, y=svr_forecast_set, mode='lines', name='SVR Forecast'))
    fig.update_layout(title=f"{stock} Stock Price Forecast for the Next {days - 1} Days")

    return fig

def calculate_stop_loss(df, percentage=5):
    stop_loss_price = df['Close'].iloc[-1] * (1 - percentage / 100)
    return stop_loss_price

def detect_false_breakouts(df):
    breakout_df = df[df['Close'] > df['Close'].shift(1)]
    return breakout_df

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(src="/assets/bull.jpg", className="header-image", style={"width": "100%", "height": "auto"}),  # Adjust size
                html.P("✴WELCOME TO STOCK VISION✴", className="start"),
                html.Div([
                    html.P("🔘Enter a stock code: "),
                    dcc.Input(id="dropdown_tickers", type="text"),
                    html.Button("Submit", id='submit', className="button")
                ], className="form"),
                dcc.DatePickerRange(
                    id='my-date-picker-range',
                    min_date_allowed=dt(1995, 8, 5),
                    max_date_allowed=dt.now(),
                    initial_visible_month=dt.now(),
                    end_date=dt.now().date()
                ),
                html.Div([
                    html.Button("📊Stock Price", className="button", id="stock"),
                    html.Button("📈📉Indicators", className="button", id="indicators"),
                    dcc.Input(id="n_days", type="text", placeholder="Enter number of days", className="input-days"),
                    html.Button("🕵️‍♀️Forecast", className="button", id="forecast"),
                    html.Button("🚨Stop Loss", className="button", id="stop-loss"),
                    html.Button("🔍False Breakout", className="button", id="false-breakout")
                ], className="buttons"),
            ],
            className="nav"
        ),
        html.Div(
            [
                html.Div([html.Img(id="logo"), html.P(id="ticker")], className="header"),
                html.Div(id="description", className="description_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content"),
                html.Div([], id="stop-loss-content"),
                html.Div([], id="false-breakout-content"),
            ],
            className="content"
        ),
    ],
    className="container"
)


@app.callback(
    [Output("description", "children"), Output("logo", "src"), Output("ticker", "children")],
    [Input("submit", "n_clicks")],
    [State("dropdown_tickers", "value")]
)
def update_data(n, val):
    if n is None:
        return (
            " ",
            "/assets/stock.png",  # Path to the local image in the assets folder
            "STOCK VISION \n - A Real-time Stock Visualizer and Forecaster"
        )
    if val is None:
        raise PreventUpdate
    ticker = yf.Ticker(val)
    inf = ticker.info
    df = pd.DataFrame().from_dict(inf, orient="index").T
    return (
        df['longBusinessSummary'].values[0],
        df['logo_url'].values[0],
        df['shortName'].values[0]
    )

@app.callback([Output("graphs-content", "children")], [Input("stock", "n_clicks"), Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')], [State("dropdown_tickers", "value")])
def stock_price(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date is not None else yf.download(val)
    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("main-content", "children")], [Input("indicators", "n_clicks"), Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n is None:
        return [""]
    if val is None:
        return [""]
    df_more = yf.download(val, str(start_date), str(end_date)) if start_date is not None else yf.download(val)
    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("forecast-content", "children")], [Input("forecast", "n_clicks")], [State("n_days", "value"), State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n is None:
        return [""]
    if val is None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("stop-loss-content", "children")], [Input("stop-loss", "n_clicks")], [State('dropdown_tickers', 'value'), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')])
def display_stop_loss(n, val, start_date, end_date):
    if n is None or val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date and end_date else yf.download(val)
    df.reset_index(inplace=True)
    stop_loss_price = calculate_stop_loss(df)
    fig = get_stock_price_fig(df)
    fig.update_layout(title="Stock Price with Stop Loss Line")
    fig.add_shape(type="line", x0=df['Date'].min(), y0=stop_loss_price, x1=df['Date'].max(), y1=stop_loss_price, line=dict(color="Red", width=2, dash="dashdot"))
    fig.add_annotation(x=df['Date'].max(), y=stop_loss_price, text=f"Stop Loss: ${stop_loss_price:.2f}", showarrow=False, yshift=10)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("false-breakout-content", "children")], [Input("false-breakout", "n_clicks")], [State('dropdown_tickers', 'value'), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')])
def display_false_breakout(n, val, start_date, end_date):
    if n is None or val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date and end_date else yf.download(val)
    df.reset_index(inplace=True)
    breakout_df = detect_false_breakouts(df)
    fig = get_stock_price_fig(df)
    fig.update_layout(title="Stock Price with False Breakouts")
    fig.add_trace(go.Scatter(x=breakout_df['Date'], y=breakout_df['Close'], mode='markers', name='False Breakout', marker=dict(color='orange', size=10)))
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run_server(debug=True)