from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import numpy as np
from datetime import date, timedelta
import plotly.graph_objs as go

def prediction(stock, n_days):
    # Load stock data
    df = yf.download(stock, period='6mo')
    df.reset_index(inplace=True)
    df['Day'] = df.index

    days = list()
    for i in range(len(df.Day)):
        days.append([i])

    # Splitting the dataset
    X = days
    Y = df[['Close']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # GridSearch for SVR model hyperparameter tuning
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    # Fit the model
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)

    # Train the best model
    best_svr.fit(x_train, y_train)

    # Make predictions on test data
    svr_predictions = best_svr.predict(x_test)

    # Calculate Metrics for SVR
    svr_mse = mean_squared_error(y_test, svr_predictions)
    svr_mae = mean_absolute_error(y_test, svr_predictions)
    svr_accuracy = 100 - np.mean(np.abs((y_test - svr_predictions) / y_test)) * 100

    print(f"SVR - MSE: {svr_mse:.2f}, MAE: {svr_mae:.2f}, Accuracy: {svr_accuracy:.2f}%")

    # Prepare future prediction days
    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + x_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    # Create Plotly figure for predicted stock prices
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=best_svr.predict(output_days),
            mode='lines+markers',
            name='Predicted Prices'
        )
    )

    fig.update_layout(
        title=f"Predicted Close Price for the Next {n_days - 1} Days",
        xaxis_title="Date",
        yaxis_title="Close Price",
    )

    return fig
