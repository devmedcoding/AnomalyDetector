import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_actual_vs_predicted(dates, actual, predicted, title, label1='Actual', label2='Predicted'):
    dates = pd.to_datetime(dates)
    """
    Plot given dataset.
    """
    logging.info(f"Plotting {title}...")
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, 'b-', label=label1)
    plt.plot(dates, predicted, 'r-', label=label2)
    plt.title(title)
    plt.legend()
    plt.show()


def get_residuals_and_anomalies(dates, actual, predicted, upper_threshold, lower_threshold):
    dates = pd.to_datetime(dates)
    residuals = actual - predicted

    anomalies_data = {
        'high': {'dates': [], 'values': [], 'size': [], 'alpha': []},
        'low': {'dates': [], 'values': [], 'size': [], 'alpha': []}
    }

    for i in range(len(residuals)):
        if residuals.iloc[i] > upper_threshold.iloc[i]:
            anomalies_data['high']['dates'].append(dates.iloc[i])
            anomalies_data['high']['values'].append(actual.iloc[i])
            anomalies_data['high']['size'].append(np.clip(30 * abs(residuals.iloc[i]) / max(abs(residuals)), 30, 100))
            anomalies_data['high']['alpha'].append(np.clip(abs(residuals.iloc[i]) / max(abs(residuals)), 0.2, 1))
        elif residuals.iloc[i] < lower_threshold.iloc[i]:
            anomalies_data['low']['dates'].append(dates.iloc[i])
            anomalies_data['low']['values'].append(actual.iloc[i])
            anomalies_data['low']['size'].append(np.clip(30 * abs(residuals.iloc[i]) / max(abs(residuals)), 30, 100))
            anomalies_data['low']['alpha'].append(np.clip(abs(residuals.iloc[i]) / max(abs(residuals)), 0.2, 1))

    return residuals, anomalies_data


def get_actual_anomalies(prophet_test_data, test_element_data):
    actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1]
    actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1]
    return actual_anomalies_dates, actual_anomalies_values


def plot_anomalies_chart(dates, actual, predicted, upper_threshold, lower_threshold, anomalies_data, actual_anomalies_dates, actual_anomalies_values):
    dates = pd.to_datetime(dates)  # Parse dates into datetime
    actual_anomalies_dates = pd.to_datetime(actual_anomalies_dates)

    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, 'b-', label='Actual')
    plt.plot(dates, predicted, 'r-', label='Predicted')
    plt.plot(dates, upper_threshold, 'g--', label='Upper Dynamic Threshold')
    plt.plot(dates, lower_threshold, 'y--', label='Lower Dynamic Threshold')

    plt.scatter(anomalies_data['high']['dates'], anomalies_data['high']['values'], color='purple',
                s=anomalies_data['high']['size'],
                label='High Anomalies', zorder=6, marker="^")
    plt.scatter(anomalies_data['low']['dates'], anomalies_data['low']['values'], color='darkblue',
                s=anomalies_data['low']['size'],
                label='Low Anomalies', zorder=6, marker="v")

    plt.scatter(actual_anomalies_dates, actual_anomalies_values, color='magenta', s=80, label='Actual Anomalies',
                zorder=5, marker="X")

    plt.legend(loc='best')
    plt.title('Time Series with Detected Anomalies')
    plt.show()


def generate_anomalies_csv(dates, actual, predicted, upper_threshold, lower_threshold, prophet_test_data, test_element_data):

    # No need to calculate residuals here
    logging.info("Generating Anomaly Comparison CSV File...")
    anomalies_df = construct_anomalies_dataframe(dates, upper_threshold, lower_threshold, actual, prophet_test_data,
                                                 test_element_data, predicted)
    anomalies_df.to_csv("anomalies.csv", index=False)


def construct_anomalies_dataframe(dates, upper_threshold, lower_threshold, actual, prophet_test_data,
                                  test_element_data, predicted):
    anomaly_dates = []
    anomaly_actual_values = []
    anomaly_detected_values = []
    anomaly_type = []

    residuals = actual - predicted

    for i in range(len(residuals)):
        if residuals.iloc[i] > upper_threshold.iloc[i]:
            anomaly_dates.append(dates.iloc[i])
            anomaly_actual_values.append(actual.iloc[i])  # Or actual.iloc[i] if they are the same
            anomaly_detected_values.append(predicted[i])
            anomaly_type.append('High Detected Anomaly')
        elif residuals.iloc[i] < lower_threshold.iloc[i]:
            anomaly_dates.append(dates.iloc[i])
            anomaly_actual_values.append(actual.iloc[i])  # Or actual.iloc[i] if they are the same
            anomaly_detected_values.append(predicted[i])
            anomaly_type.append('Low Detected Anomaly')

    actual_anomalies_dates = prophet_test_data['ds'][5:][test_element_data['Label'][5:] == 1].values
    actual_anomalies_values = prophet_test_data['y'][5:][test_element_data['Label'][5:] == 1].values

    for i in range(len(actual_anomalies_dates)):
        anomaly_dates.append(actual_anomalies_dates[i])
        anomaly_actual_values.append(actual_anomalies_values[i])
        anomaly_detected_values.append(None)  # No detected value for actual anomalies
        anomaly_type.append('True Anomaly')

    anomalies_df = pd.DataFrame({
        'Date': anomaly_dates,
        'Actual Value': anomaly_actual_values,
        'Detected Value': anomaly_detected_values,
        'Anomaly Type': anomaly_type
    })

    return anomalies_df
