import logging
import pandas as pd


def calculate_dynamic_threshold(residuals, window_size=144, timestamps=None):
    logging.info("Calculating Dynamic Threshold [anomaly_detection.py --> calculate_dynamic_threshold]...")

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


def calculate_residual_zscore(residuals, window=144):
    logging.info("Calculating Residual Z Score [anomaly_detection.py --> calculate_residual_zscore]...")

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
    logging.info("Smoothing Residuals [anomaly_detection.py --> smooth_residuals]...")

    """
    Smooth residuals using rolling mean.
    :param residuals: Residuals of the time series.
    :param window: Window size for rolling mean.
    :return: smoothed residuals.
    """
    return residuals.rolling(window=window).mean().fillna(0)
