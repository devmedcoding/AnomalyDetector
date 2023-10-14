import logging
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_and_run_lstm(residuals_train, look_back=288, epochs=50):
    logging.info("Building and Running LSTM Model [model_lstm.py --> build_and_run_lstm]...")

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
        LSTM(50, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
             use_bias=True,
             input_shape=(look_back, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    # Train the model
    lstm_model.fit(train_generator, epochs=epochs, verbose=1)

    return lstm_model, scaler
