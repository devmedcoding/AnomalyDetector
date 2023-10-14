# Anomaly Detector using Prophet and LSTM
## Overview
This project implements an anomaly detection system using two models: the Prophet model developed by Facebook and an LSTM (Long Short-Term Memory) neural network model. By leveraging the strengths of both time-series forecasting and deep learning, this detector aims to identify anomalies efficiently and accurately.

<br>

## Features
- **Prophet Model**: A robust procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
<!-- -->
- **LSTM Model**: A type of recurrent neural network (RNN) architecture that excels in learning and remembering over long sequences and is quite effective for time series forecasting.

<br>

## Prerequisites
- Python 3.x
- TensorFlow
- Prophet
- pandas
- numpy
- matplotlib (for visualization, if you're using it)

<br><br>

## Installation & Setup

### Setting up the Conda Environment:
> **Note**: After installing Anaconda or Miniconda, open the Anaconda Prompt or terminal where conda is accessible to execute the following commands.

&nbsp;
1. First, make sure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) installed.


2. Navigate to the project directory where the **'requirements.txt'** file is located:
   ```
   cd path/to/your/project_directory
   ```


3. Create a new Conda environment. Replace "ENV_NAME" with any name you'd like for the environment, and adjust the Python version if needed:
   ```
   conda create --name ENV_NAME python=3.9
   ```


4. Activate the Conda environment:
   ```
   conda activate ENV_NAME
   ```
<br>

### Installing Dependencies:
   
   > **Note**: Ensure all the following commands are executed within the activated Conda environment.
   <br> 
1. For enhanced performance with the LSTM model, especially if you have a GPU, consider installing both CUDA and cuDNN. These tools optimize neural network computations, making the LSTM model run faster. If you don't have a GPU, you can skip this step:
   ```
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```


2. To ensure you have the latest version of pip, which can help in avoiding potential package installation issues and accessing the most recent package releases, execute the following command:
   ```
   pip install --upgrade pip
   ```


3. To set up your environment with all the necessary dependencies, install the packages listed in the **'requirements.txt'** file. This ensures that your setup matches the expected versions and packages for the project, minimizing compatibility issues:
   ```
   pip install -r requirements.txt
   ```

<br><br>

## Setting Up Your IDE:

### General:
Ensure your chosen Integrated Development Environment (IDE) is using the Conda environment you've set up as its Python interpreter.
<br>
### For PyCharm Users:
1. Open the "Preferences" or "Settings" dialog.
   
2. Navigate to "Project: [Your_Project_Name] > Python Interpreter."
   
3. In the Python interpreter dropdown, select the Conda environment you've created. If it's not listed:
    - Click on the gear icon or "Add Interpreter" button on the top right.
      
    - Choose "Add."
      
    - Select "Conda Environment" on the left, and ensure "Existing environment" is selected.
      
    - Navigate to and select the Python executable from your Conda environment (usually found in envs/ENV_NAME/bin/python for Unix-like systems or envs\ENV_NAME\python.exe for Windows).
      
    - Click "OK."

<br>

### For VSCode Users:
1. Open the Command Palette (Ctrl+Shift+P).
   
2. Search and select "Python: Select Interpreter."
   
3. Choose the Conda environment you've created from the list.

<br>

### For Jupyter Notebook Users:
1. Launch Jupyter Notebook.
   
2. In the notebook interface, click on the Kernel menu.
   
3. Select "Change kernel" and choose the Conda environment you've set up.

<br>

> **Note: For other IDEs not listed above, please refer to the respective IDE's documentation on how to set a Python interpreter.**

<br><br>

## Usage

### Preparing Your Data:

1. Locate the **'data'** folder in the project directory. Inside this, you'll find a raw folder.
2. Move your raw data files into the raw folder.
3. In the **'data_preprocessing.py'** script, update the file names to match the ones you've added.
   
   The relevant constants to change are:
   ```
   RAW_TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "YOUR_TRAIN_FILE_NAME.csv")
   RAW_TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "YOUR_TEST_FILE_NAME.csv")
   ```

<br><br>

## Conclusion

The Anomaly Detector utilizing both the Prophet model and LSTM offers a comprehensive approach to identifying discrepancies in time-series data. By integrating the precision of time-series forecasting with the deep learning capabilities of LSTM, this system serves as a powerful tool for diverse applications. Whether you're new to anomaly detection or seeking an efficient model to improve your existing workflows, this project aims to provide an accessible and accurate solution.
