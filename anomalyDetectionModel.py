import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Setup logging
logging.basicConfig(level=logging.INFO)


def load_and_preprocess_data():
    logging.info("Starting load_and_preprocess_data...")

    """
    Load datasets and preprocess data.
    :return: train, valid, df_test
    """
    # Load the datasets
    df_train = pd.read_csv("CallAttempts_Alex.csv")
    df_test = pd.read_csv("CallAttempts_Anomaly_Alex.csv")

    # Filter out -1 and null values from the 'Call_Attempts' column
    df_train = df_train[df_train['Call_Attempts'] != -1]
    df_train = df_train[~df_train['Call_Attempts'].isnull()]

    df_test = df_test[df_test['Call_Attempts'] != -1]
    df_test = df_test[~df_test['Call_Attempts'].isnull()]

    # Convert the STARTTIMESTAMP to datetime format
    df_train['STARTTIMESTAMP'] = pd.to_datetime(df_train['STARTTIMESTAMP'])
    df_test['STARTTIMESTAMP'] = pd.to_datetime(df_test['STARTTIMESTAMP'])

    # Resampling and interpolation
    df_train = resample_by_element(df_train)
    df_test = resample_by_element(df_test)

    # Drop any rows with NaN values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Sort by timestamp
    df_train.sort_values(by="STARTTIMESTAMP", inplace=True)
    df_test.sort_values(by="STARTTIMESTAMP", inplace=True)

    # Splitting data into train and validation sets (last 2 weeks for validation)
    train = df_train.iloc[:-2 * 7 * 24 * 12]
    valid = df_train.iloc[-2 * 7 * 24 * 12:]

    return train, valid, df_test


def resample_by_element(df):
    logging.info("Starting resample_by_element...")

    """
    Resamples and interpolates for each unique element.
    :param df: DataFrame
    :return: Resampled DataFrame
    """
    elements = df['ELEMENT'].unique()
    resampled_dfs = []
    for elem in elements:
        temp = df[df['ELEMENT'] == elem].drop_duplicates(subset=['STARTTIMESTAMP']).copy()
        temp.set_index("STARTTIMESTAMP", inplace=True)
        numeric_cols = temp.select_dtypes(include=['number']).columns
        temp[numeric_cols] = temp[numeric_cols].resample('5T').interpolate(method='time')
        resampled_dfs.append(temp.reset_index())
        logging.info(f"NaN Values for element {elem}:")
        print(temp.isnull().sum())
    return pd.concat(resampled_dfs, axis=0)


def run_prophet(train_data):
    logging.info("Starting run_prophet...")

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


def build_and_run_lstm(residuals_train, look_back=288, epochs=50):
    logging.info("Starting build_and_run_lstm...")

    """
    Build and run the LSTM model.
    :param residuals_train: Residuals from Prophet model on training data
    :param look_back: Number of previous time steps to use for prediction
    :param epochs: Number of training epochs
    :return: lstm_model, scaler
    """

    # Scale the residuals
    scaler = MinMaxScaler()
    residuals_train_scaled = scaler.fit_transform(residuals_train.values.reshape(-1, 1))

    # Prepare data sequences
    train_generator = TimeseriesGenerator(residuals_train_scaled, residuals_train_scaled, length=look_back,
                                          batch_size=10)

    # Design the LSTM model
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(50, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True,
             input_shape=(look_back, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    # Train the model
    lstm_model.fit(train_generator, epochs=epochs, verbose=1)

    return lstm_model, scaler


def calculate_residual_zscore(residuals, window=144):
    """
    Calculate z-score for residuals.
    :param residuals: Residuals of the time series.
    :param window: Window size for rolling mean and standard deviation.
    :return: z-scores.
    """
    mean_rolling = residuals.rolling(window=window).mean()
    std_rolling = residuals.rolling(window=window).std()

    z_scores = (residuals - mean_rolling) / std_rolling
    return z_scores


def smooth_residuals(residuals, window=1):
    """
    Smooth residuals using rolling mean.
    :param residuals: Residuals of the time series.
    :param window: Window size for rolling mean.
    :return: smoothed residuals.
    """
    return residuals.rolling(window=window).mean().fillna(0)


def calculate_dynamic_threshold(residuals, window_size=144, timestamps=None):
    logging.info("Starting calculate_dynamic_threshold...")

    """
    Calculate dynamic threshold using Rolling Mean and Rolling Standard Deviation.
    :param residuals: The residuals for which to calculate the threshold.
    :param window_size: Size of the rolling window.
    :return: Rolling Mean, Upper threshold, Lower threshold
    """

    rolling_mean = residuals.rolling(window=window_size, min_periods=1).mean()
    rolling_std = residuals.rolling(window=window_size, min_periods=1).std()

    # Default threshold multiplier (for non-morning hours)
    threshold_multiplier_default = 4.5
    # Adaptive threshold multiplier for the morning window
    threshold_multiplier_morning = 4

    if timestamps is not None:
        upper_threshold = []
        lower_threshold = []
        for idx, timestamp in enumerate(timestamps):
            timestamp = pd.to_datetime(timestamp)  # Convert to pandas.Timestamp
            hour = timestamp.hour
            if 6 <= hour < 10:
                threshold_multiplier = threshold_multiplier_morning
            else:
                threshold_multiplier = threshold_multiplier_default

            upper_threshold.append(rolling_mean.iloc[idx] + threshold_multiplier * rolling_std.iloc[idx])
            lower_threshold.append(rolling_mean.iloc[idx] - threshold_multiplier * rolling_std.iloc[idx])

        upper_threshold = pd.Series(upper_threshold, index=timestamps)
        lower_threshold = pd.Series(lower_threshold, index=timestamps)
    else:
        upper_threshold = rolling_mean + threshold_multiplier_default * rolling_std
        lower_threshold = rolling_mean - threshold_multiplier_default * rolling_std

    return rolling_mean, upper_threshold, lower_threshold


def plot_with_anomalies(dates, actual, predicted, upper_threshold, lower_threshold, prophet_test_data,
                        test_element_data):
    logging.info("Starting plot_with_anomalies...")

    """
    Plot time series with anomalies.
    :param dates: List of datetime objects.
    :param actual: Actual values.
    :param predicted: Predicted values.
    :param upper_threshold: Dynamic threshold.
    """
    residuals = actual - predicted

    # Lists for anomalies and their corresponding dates
    anomalies_high_dates = []
    anomalies_high_values = []

    anomalies_low_dates = []
    anomalies_low_values = []

    # Lists for size and alpha values corresponding to anomalies
    size_values_high = []
    alpha_values_high = []

    size_values_low = []
    alpha_values_low = []

    for i in range(len(residuals)):
        if residuals.iloc[i] > upper_threshold.iloc[i]:
            anomalies_high_dates.append(dates.iloc[i])
            anomalies_high_values.append(actual.iloc[i])
            size_values_high.append(np.clip(30 * abs(residuals.iloc[i]) / max(abs(residuals)), 30, 100))
            alpha_values_high.append(np.clip(abs(residuals.iloc[i]) / max(abs(residuals)), 0.2, 1))
        elif residuals.iloc[i] < lower_threshold.iloc[i]:
            anomalies_low_dates.append(dates.iloc[i])
            anomalies_low_values.append(actual.iloc[i])
            size_values_low.append(np.clip(30 * abs(residuals.iloc[i]) / max(abs(residuals)), 30, 100))
            alpha_values_low.append(np.clip(abs(residuals.iloc[i]) / max(abs(residuals)), 0.2, 1))

    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, 'b-', label='Actual')
    plt.plot(dates, predicted, 'r-', label='Predicted')
    plt.plot(dates, upper_threshold, 'g--', label='Upper Dynamic Threshold')
    plt.plot(dates, lower_threshold, 'y--', label='Lower Dynamic Threshold')

    plt.scatter(anomalies_high_dates, anomalies_high_values, color='darkred', s=size_values_high,
                label='High Anomalies', zorder=6, marker="^")
    plt.scatter(anomalies_low_dates, anomalies_low_values, color='darkblue', s=size_values_low,
                label='Low Anomalies', zorder=6, marker="v")

    # Extracting actual anomalies from the test set
    actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1]
    actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1]

    # Use a different zorder and marker style for true anomalies
    plt.scatter(actual_anomalies_dates, actual_anomalies_values, color='magenta', s=80,
                label='Actual Anomalies', zorder=5, marker="X")

    plt.legend(loc='best')
    plt.title('Time Series with Detected Anomalies')
    plt.show()

    # Counting anomalies
    actual_anomalies_count = (test_element_data['Label'][5:] == 1).sum()
    detected_anomalies_count = len(anomalies_high_dates) + len(anomalies_low_dates)

    generate_anomalies_csv(dates, actual, predicted, upper_threshold, lower_threshold, prophet_test_data,
                           test_element_data)

    print(f"Actual anomalies: {actual_anomalies_count}")
    print(f"Detected anomalies: {detected_anomalies_count}")


def generate_anomalies_csv(dates, actual, predicted, upper_threshold, lower_threshold, prophet_test_data, test_element_data):
    residuals = actual - predicted

    # Lists to store anomalies data
    anomaly_dates = []
    anomaly_actual_values = []
    anomaly_detected_values = []
    anomaly_type = []

    for i in range(len(residuals)):
        if residuals.iloc[i] > upper_threshold.iloc[i]:
            anomaly_dates.append(dates.iloc[i])
            anomaly_actual_values.append(actual.iloc[i])
            anomaly_detected_values.append(predicted.iloc[i])
            anomaly_type.append('High Detected Anomaly')
        elif residuals.iloc[i] < lower_threshold.iloc[i]:
            anomaly_dates.append(dates.iloc[i])
            anomaly_actual_values.append(actual.iloc[i])
            anomaly_detected_values.append(predicted.iloc[i])
            anomaly_type.append('Low Detected Anomaly')

    actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1].values
    actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1].values

    for i in range(len(actual_anomalies_dates)):
        anomaly_dates.append(actual_anomalies_dates[i])
        anomaly_actual_values.append(actual_anomalies_values[i])
        anomaly_detected_values.append(None)  # No detected value for actual anomalies
        anomaly_type.append('True Anomaly')

    # Construct a DataFrame and save to CSV
    anomalies_df = pd.DataFrame({
        'Date': anomaly_dates,
        'Actual Value': anomaly_actual_values,
        'Detected Value': anomaly_detected_values,
        'Anomaly Type': anomaly_type
    })

    anomalies_df.to_csv("anomalies.csv", index=False)


def main():
    # Load and preprocess data
    train, valid, df_test = load_and_preprocess_data()

    # Run Prophet model
    prophet_model, forecast_train, residuals_train, prophet_data = run_prophet(train[train['ELEMENT'] == 'ELEMENT1'])

    # Plotting for the training dataset
    plt.figure(figsize=(15, 6))
    plt.plot(prophet_data['ds'], prophet_data['y'], label='Actual')
    plt.plot(prophet_data['ds'], forecast_train['yhat'], label='Predicted by Prophet')
    plt.title('Training Data: Actual vs Prophet Predictions for ELEMENT1')
    plt.legend()
    plt.show()

    # Extract element data from test dataset
    test_element_data = df_test[df_test['ELEMENT'] == 'ELEMENT1']
    prophet_test_data = test_element_data[['STARTTIMESTAMP', 'Call_Attempts']]
    prophet_test_data.columns = ['ds', 'y']

    # Predict with Prophet on the testing data
    forecast_test = prophet_model.predict(prophet_test_data)
    residuals_test = prophet_test_data['y'].values - forecast_test['yhat'].values

    # Plotting for the test dataset
    plt.figure(figsize=(15, 6))
    plt.plot(prophet_test_data['ds'], prophet_test_data['y'], label='Actual')
    plt.plot(prophet_test_data['ds'], forecast_test['yhat'], label='Predicted by Prophet')
    plt.title('Test Data: Actual vs Prophet Predictions for ELEMENT1')
    plt.legend()
    plt.show()

    # Smooth the residuals before feeding to LSTM for both training and testing
    smoothed_residuals_train = smooth_residuals(pd.Series(residuals_train), window=1)
    smoothed_residuals_test = smooth_residuals(pd.Series(residuals_test), window=1)

    # Build and run LSTM
    lstm_model, scaler = build_and_run_lstm(smoothed_residuals_train)

    # Scale the smoothed residuals
    residuals_test_scaled = scaler.transform(smoothed_residuals_test.values.reshape(-1, 1))

    # Predict residuals using LSTM
    test_generator = TimeseriesGenerator(residuals_test_scaled, residuals_test_scaled, length=5, batch_size=1)
    predicted_residuals_scaled = lstm_model.predict(test_generator)
    predicted_residuals = scaler.inverse_transform(predicted_residuals_scaled)

    # Add predicted residuals to Prophet's predictions to get the final predictions
    final_predictions = forecast_test['yhat'].values[5:] + predicted_residuals.ravel()

    # Plotting for the test dataset with LSTM enhancements
    plt.figure(figsize=(15, 6))
    plt.plot(prophet_test_data['ds'][5:], prophet_test_data['y'][5:], label='Actual')
    plt.plot(prophet_test_data['ds'][5:], final_predictions, label='Predicted by Prophet + LSTM')
    plt.title('Test Data: Actual vs Combined Predictions for ELEMENT1')
    plt.legend()
    plt.show()

    residuals = pd.Series(prophet_test_data['y'][5:].values - final_predictions)

    # Extracting actual anomalies from the test set
    actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1].values
    actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1].values

    # Detect anomalies using Z-Scores
    residual_z_scores = calculate_residual_zscore(residuals, window=12)  # Adjusted the window for intraday sensitivity
    anomalies = residuals[np.abs(residual_z_scores) > 5]

    # Plot residuals
    plt.figure(figsize=(15, 7))
    plt.plot(prophet_test_data['ds'][5:], residuals, label="Residuals")
    plt.scatter(prophet_test_data['ds'][5:].iloc[anomalies.index], anomalies, color='red', s=100,
                label='Detected Anomalies', marker='o', edgecolors='black')
    plt.scatter(actual_anomalies_dates, actual_anomalies_values, color='purple', s=120, label='True Anomalies',
                marker='X',
                edgecolors='black')
    plt.axhline(0, color="grey", linestyle="--")
    plt.legend()
    plt.title("Residuals Over Time with Anomalies Highlighted")
    plt.show()

    rolling_mean, upper_threshold, lower_threshold = calculate_dynamic_threshold(residuals,
                                                                                 timestamps=prophet_test_data['ds'][
                                                                                            5:].values)

    plot_with_anomalies(prophet_test_data['ds'][5:], prophet_test_data['y'][5:], final_predictions, upper_threshold,
                        lower_threshold, prophet_test_data, test_element_data)

    # Evaluate the predictions using MAE and RMSE
    # MAE - Average magnitude of the errors between predicted and observed values
    mae = mean_absolute_error(prophet_test_data['y'][5:], final_predictions)
    rmse = mean_squared_error(prophet_test_data['y'][5:], final_predictions, squared=False)
    logging.info(f"Mean Absolute Error (MAE): {mae}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse}")


if __name__ == "__main__":
    main()
