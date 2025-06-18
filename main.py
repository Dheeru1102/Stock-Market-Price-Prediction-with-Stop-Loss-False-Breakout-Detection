import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dash.exceptions import PreventUpdate
import numpy as np


app = dash.Dash(__name__)
# Function to create stock price figure with axis labels
def get_stock_price_fig(df):
    df['Date'] = pd.to_datetime(df['Date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Opening Price', line=dict(color='blue')))
    fig = px.line(df, x='Date', y=['Open', 'Close'], labels={'value': 'Price (USD)', 'Date': 'Date'})
    fig.update_xaxes(tickformat='%b %d, %Y')  # Format date as Month Day, Year
    fig.show()
    fig.update_layout(xaxis=dict(type='category'))
    fig.update_layout(
        title="Stock Price (Close & Open) Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor='lightgray',
        paper_bgcolor='white'
    )
    return fig

# Function to create EMA-20 figure with axis labels
def get_more(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='brown')))
    fig.update_layout(
        title="Exponential Moving Average (EMA-20) Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor='lightgray',
        paper_bgcolor='white'
    )
    return fig

# Function to create prediction figure with axis labels
def prediction(stock, days):
    df = yf.download(stock)
    df['Date'] = pd.to_datetime(df.index)
    df['Prediction'] = df[['Close']].shift(-days)
    X = np.array(df[['Close']])
    X = X[:-days]
    y = np.array(df['Prediction'])
    y = y[:-days]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    forecast_set = model.predict(np.array(df[['Close']])[-days:])

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    accuracy = 100 - mape

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")

    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_set})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='green')))
    fig.update_layout(
        title=f"{stock} Stock Price Forecast for the Next {days - 1} Days",
        xaxis_title="Date",
        yaxis_title="Predicted Price (USD)",
        plot_bgcolor='lightgray',
        paper_bgcolor='white'
    )
    return fig

# Function to calculate stop loss price
def calculate_stop_loss(df, percentage=5):
    stop_loss_price = df['Close'].iloc[-1] * (1 - percentage / 100)
    return stop_loss_price

# Function to detect false breakouts
def detect_false_breakouts(df):
    breakout_df = df[df['Close'] > df['Close'].shift(1)]
    return breakout_df

# Layout definition
app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(src="/assets/bull.jpg", className="header-image", style={"width": "100%", "height": "auto"}),
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

# Callbacks
@app.callback(
    [Output("description", "children"), Output("logo", "src"), Output("ticker", "children")],
    [Input("submit", "n_clicks")],
    [State("dropdown_tickers", "value")]
)
def update_data(n, val):
    if n is None:
        return (
            " ",
            "/assets/stock.png",
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
        return [" "]
    if val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date is not None else yf.download(val)
    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("main-content", "children")], [Input("indicators", "n_clicks"), Input('my-date-picker-range', 'start_date'), Input('my-date-picker-range', 'end_date')], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n is None:
        return [" "]
    if val is None:
        return [" "]
    df_more = yf.download(val, str(start_date), str(end_date)) if start_date is not None else yf.download(val)
    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

@app.callback([Output("forecast-content", "children")], [Input("forecast", "n_clicks")], [State("n_days", "value"), State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n is None:
        return [" "]
    if val is None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

@app.callback(
    [Output("stop-loss-content", "children")],
    [Input("stop-loss", "n_clicks")],
    [State('dropdown_tickers', 'value'), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')]
)
def display_stop_loss(n, val, start_date, end_date):
    if n is None or val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date and end_date else yf.download(val)
    df.reset_index(inplace=True)
    stop_loss_price = calculate_stop_loss(df)

    # Ensure the stop loss price is within the y-axis range for better visibility
    y_min, y_max = df['Close'].min(), df['Close'].max()
    if stop_loss_price < y_min or stop_loss_price > y_max:
        return [html.P("Stop loss price is outside the visible range of the graph.")]

    fig = get_stock_price_fig(df)
    fig.add_hline(
        y=stop_loss_price,
        line_dash="dash",
        line_color="magenta"
    )
    # Adding annotation separately for better control
    fig.add_annotation(
        x=df['Date'].iloc[-1],  # Last date on the x-axis
        y=stop_loss_price,
        text=f"Stop Loss: ${stop_loss_price:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,  # Adjust vertical position of the text
        bgcolor="white",
        bordercolor="black"
    )

    return [dcc.Graph(figure=fig)]



@app.callback([Output("false-breakout-content", "children")], [Input("false-breakout", "n_clicks")], [State('dropdown_tickers', 'value'), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')])
def display_false_breakouts(n, val, start_date, end_date):
    if n is None or val is None:
        raise PreventUpdate
    df = yf.download(val, str(start_date), str(end_date)) if start_date and end_date else yf.download(val)
    df.reset_index(inplace=True)
    breakout_df = detect_false_breakouts(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=breakout_df['Date'], y=breakout_df['Close'], mode='markers', name='Breakouts', marker=dict(color='red', size=8)))
    fig.update_layout(
        title="Breakout Analysis",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor='lightgray',
        paper_bgcolor='white'
    )
    return [dcc.Graph(figure=fig)]

if __name__ == "__main__":
    app.run_server(debug=True)