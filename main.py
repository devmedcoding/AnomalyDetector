import tensorflow as tf
import logging
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from src import data_preprocessing
from src import model_prophet
from src import model_lstm
from src import anomaly_detection
from src import visualization

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def main():
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data.")
        train, valid, df_test = data_preprocessing.load_and_preprocess_data()

        # Filter data for specific element and run Prophet model
        logging.info("Running Prophet model for ELEMENT1.")
        prophet_model, forecast_train, residuals_train, prophet_data = model_prophet.run_prophet(
            train[train['ELEMENT'] == 'ELEMENT1'])

        # Visualize predictions on the training dataset
        logging.info("Visualizing Prophet predictions on training data.")
        visualization.visualize_actual_vs_predicted(prophet_data['ds'], prophet_data['y'], forecast_train['yhat'],
                                                    'Training Data: Actual vs Prophet Predictions for ELEMENT1')

        # Extract and preprocess test dataset for predictions
        test_element_data = df_test[df_test['ELEMENT'] == 'ELEMENT1']
        prophet_test_data = test_element_data[['STARTTIMESTAMP', 'Call_Attempts']]
        prophet_test_data.columns = ['ds', 'y']

        # Use Prophet model to make predictions on test data
        logging.info("Running Prophet model on test data.")
        forecast_test = prophet_model.predict(prophet_test_data)
        residuals_test = prophet_test_data['y'].values - forecast_test['yhat'].values

        # Visualize predictions on test data
        logging.info("Visualizing Prophet predictions on test data.")
        visualization.visualize_actual_vs_predicted(prophet_test_data['ds'], prophet_test_data['y'],
                                                    forecast_test['yhat'],
                                                    'Test Data: Actual vs Prophet Predictions for ELEMENT1')

        # Smooth the residuals (difference between actuals and predictions)
        logging.info("Smoothing residuals for LSTM model.")
        smoothed_residuals_train = anomaly_detection.smooth_residuals(pd.Series(residuals_train), window=1)
        smoothed_residuals_test = anomaly_detection.smooth_residuals(pd.Series(residuals_test), window=1)

        # Build LSTM model and make predictions on smoothed residuals
        logging.info("Building and running LSTM model.")
        lstm_model, scaler = model_lstm.build_and_run_lstm(smoothed_residuals_train)
        residuals_test_scaled = scaler.transform(smoothed_residuals_test.values.reshape(-1, 1))
        test_generator = TimeseriesGenerator(residuals_test_scaled, residuals_test_scaled, length=5, batch_size=1)
        predicted_residuals_scaled = lstm_model.predict(test_generator)
        predicted_residuals = scaler.inverse_transform(predicted_residuals_scaled)

        # Combine Prophet's and LSTM's predictions for final results
        logging.info("Combining Prophet and LSTM predictions.")
        final_predictions = forecast_test['yhat'].values[5:] + predicted_residuals.ravel()

        # Visualize combined predictions on test data
        logging.info("Visualizing combined predictions on test data.")
        visualization.visualize_actual_vs_predicted(prophet_test_data['ds'][5:], prophet_test_data['y'][5:],
                                                    final_predictions,
                                                    'Test Data: Actual vs Combined Predictions for ELEMENT1')

        # Calculate residuals and identify anomalies
        logging.info("Calculating residuals and identifying anomalies.")
        residuals = pd.Series(prophet_test_data['y'][5:].values - final_predictions)
        actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1].values
        actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1].values
        # residual_z_scores = anomaly_detection.calculate_residual_zscore(residuals, window=12)
        # anomalies = residuals[np.abs(residual_z_scores) > 5]

        # Calculate dynamic thresholds for anomaly detection
        logging.info("Calculating dynamic thresholds for anomaly detection.")
        rolling_mean, upper_threshold, lower_threshold = anomaly_detection.calculate_dynamic_threshold(residuals,
                                                                                                       timestamps=
                                                                                                       prophet_test_data[
                                                                                                           'ds'][
                                                                                                       5:].values)

        # First get anomalies_data using the provided function:
        residuals, anomalies_data = visualization.get_residuals_and_anomalies(prophet_test_data['ds'][5:],
                                                                              prophet_test_data['y'][5:],
                                                                              final_predictions, upper_threshold,
                                                                              lower_threshold)

        # Now plot using the correct function:
        visualization.plot_anomalies_chart(prophet_test_data['ds'][5:], prophet_test_data['y'][5:], final_predictions,
                                           upper_threshold, lower_threshold, anomalies_data, actual_anomalies_dates,
                                           actual_anomalies_values)

        visualization.get_residuals_and_anomalies(prophet_test_data['ds'][5:], prophet_test_data['y'][5:],
                                                  final_predictions,
                                                  upper_threshold, lower_threshold)
        """
        # Evaluate the combined model's predictions
        logging.info("Evaluating model's predictions.")
        mae = mean_absolute_error(prophet_test_data['y'][5:], final_predictions)
        rmse = mean_squared_error(prophet_test_data['y'][5:], final_predictions, squared=False)
        logging.info(f"Mean Absolute Error (MAE): {mae}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse}")
        """

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename='logs/anomaly_detector_log.log', level=logging.INFO, format='%(asctime)s:%('
                                                                                             'levelname)s:%(message)s')
    main()
