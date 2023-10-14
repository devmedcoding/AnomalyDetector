import logging
from prophet import Prophet


def run_prophet(train_data):
    logging.info("Running Prophet Model [model_prophet.py --> run_prophet]...")

    """
    Executes the Prophet model and returns predictions and residuals.
    :param train_data: Training data
    :return: prophet_model, forecast_train, residuals_train, prophet_data
    """

    prophet_data = train_data[['STARTTIMESTAMP', 'Call_Attempts']]
    prophet_data.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=False)

    # Add custom daily seasonality to capture the 5-minute intervals pattern
    prophet_model.add_seasonality(name='five_min_intraday', period=288, fourier_order=8)
    prophet_model.add_seasonality(name='hourly', period=1 / 24, fourier_order=8)

    prophet_model.fit(prophet_data)

    # Predict on the training data to get the fitted values
    forecast_train = prophet_model.predict(prophet_data)

    # Extract the residuals
    residuals_train = prophet_data['y'].values - forecast_train['yhat'].values

    return prophet_model, forecast_train, residuals_train, prophet_data
