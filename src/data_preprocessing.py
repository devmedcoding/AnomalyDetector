import logging
import pandas as pd
import os
from sklearn.impute import SimpleImputer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for File Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
RAW_TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "CallAttempts_Alex.csv")
RAW_TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "CallAttempts_Anomaly_Alex.csv")
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "CallAttempts_Alex_Processed.csv")
PROCESSED_VALID_PATH = os.path.join(BASE_DIR, "data", "processed", "CallAttempts_Alex_Valid_Processed.csv")
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "CallAttempts_Anomaly_Alex_Processed.csv")


def ensure_directory_structure():
    """Ensure that the required directories exist, and if not, create them."""
    for dir_path in [RAW_TRAIN_PATH, RAW_TEST_PATH, PROCESSED_TRAIN_PATH, PROCESSED_VALID_PATH, PROCESSED_TEST_PATH]:
        directory = os.path.dirname(dir_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


def load_raw_data(filepath):
    """Load data from the given filepath. Raises error if file doesn't exist."""
    if os.path.exists(filepath):
        logging.info(f"Loading data from {filepath}.")
        return pd.read_csv(filepath)
    else:
        logging.error(f"File {filepath} not found!")
        raise FileNotFoundError(f"{filepath} does not exist.")


def model_based_imputation(df, strategy='mean'):
    """Impute missing values based on the given strategy (default is mean).

    Parameters:
    - df: Input DataFrame with potential missing values.
    - strategy: Strategy used for imputation ('mean', 'median', etc.).

    Returns:
    - DataFrame after imputation.
    """
    logging.info(f"Applying {strategy} imputation.")

    # Only select the numeric columns (in this case only 'Call_Attempts')
    numeric_df = df[['Call_Attempts']]

    imputer = SimpleImputer(strategy=strategy)

    # Apply the imputation
    imputed_values = imputer.fit_transform(numeric_df)

    # Replace the 'Call_Attempts' column in the original df with the imputed values
    df['Call_Attempts'] = imputed_values

    return df


def load_and_preprocess_data():
    """Main data processing function. Loads raw data, preprocesses, and returns train, validation, and test data."""
    logging.info("Starting the data loading and preprocessing process...")

    # Check if processed data already exists
    if all(map(os.path.exists, [PROCESSED_TRAIN_PATH, PROCESSED_VALID_PATH, PROCESSED_TEST_PATH])):
        logging.info("Processed data found. Loading it directly.")
        return map(pd.read_csv, [PROCESSED_TRAIN_PATH, PROCESSED_VALID_PATH, PROCESSED_TEST_PATH])

    df_train = load_raw_data(RAW_TRAIN_PATH)
    df_test = load_raw_data(RAW_TEST_PATH)

    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    train_data, valid_data = split_train_valid(df_train)

    save_processed_data(train_data, valid_data, df_test)

    return train_data, valid_data, df_test


def preprocess_data(df):
    """Perform a series of preprocessing steps on the input DataFrame and return the processed DataFrame."""
    logging.info("Starting preprocessing on data...")
    df = filter_invalid_calls(df)
    df = convert_to_datetime(df)
    df = resample_by_element(df)
    df = model_based_imputation(df)
    df.sort_values(by="STARTTIMESTAMP", inplace=True)
    return df


def filter_invalid_calls(df):
    """Remove rows with invalid Call_Attempts from the DataFrame."""
    logging.info("Filtering invalid calls...")
    return df[(df['Call_Attempts'] != -1) & df['Call_Attempts'].notnull()]


def convert_to_datetime(df):
    """Convert the STARTTIMESTAMP column to datetime format."""
    logging.info("Converting STARTTIMESTAMP to datetime format.")
    df = df.copy()
    df['STARTTIMESTAMP'] = pd.to_datetime(df['STARTTIMESTAMP'])
    return df


def resample_by_element(df):
    """Resample and interpolate data for each unique element at 5-minute intervals."""
    logging.info("Resampling and interpolating data by element.")
    elements = df['ELEMENT'].unique()
    resampled_dfs = []
    for elem in elements:
        temp_df = df[df['ELEMENT'] == elem]
        temp_df.set_index("STARTTIMESTAMP", inplace=True)
        temp_df = temp_df.resample('5T').interpolate(method='time')
        resampled_dfs.append(temp_df.reset_index())
    return pd.concat(resampled_dfs, axis=0)


def split_train_valid(df):
    """Split the input DataFrame into train and validation datasets."""
    logging.info("Splitting data into training and validation sets.")
    train_data = df.iloc[:-2 * 7 * 24 * 12]
    valid_data = df.iloc[-2 * 7 * 24 * 12:]
    return train_data, valid_data


def save_processed_data(train_data, valid_data, test_data):
    """Save processed data to appropriate paths."""
    logging.info("Saving processed data.")
    train_data.to_csv(PROCESSED_TRAIN_PATH, index=False)
    valid_data.to_csv(PROCESSED_VALID_PATH, index=False)
    test_data.to_csv(PROCESSED_TEST_PATH, index=False)
    logging.info("Processed data saved successfully.")


if __name__ == '__main__':
    ensure_directory_structure()
    train, valid, test = load_and_preprocess_data()
