"""
DATS 6103 - Final Project - Team 4

This script performs data preprocessing, feature engineering, model selection, and evaluation for predicting customer churn for a telecom company.

The dataset 1 used is ... dataset from ... (https://).
The dataset 2 used is ... dataset from ... (https://).

The project team consists of:
- Daniel Felberg
- Ei Tanaka
- Ishani Makwana
- Tharaka Maddineni 

Usage:
    Run the script in a Python environment with the necessary libraries installed. The script reads the dataset from the 'data' directory and outputs evaluation metrics and visualizations in the 'output' directory.

Output:
    - Metrics.csv: CSV file containing evaluation metrics for each model
    - Feature_Importances.png: bar chart showing feature importances for the selected model
    - Confusion_Matrix.png: confusion matrix for the selected model
    - ROC_Curve.png: ROC curve for the selected model

Note: this script requires scikit-learn version ... and pandas version ....
"""
#%%[markdown]
# # Title
# Team Member: 1, 2, 3, 4

# ## Introduction
#%%
# Set up library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Import data set
import requests
import io
import zipfile

#%%
# Import data sets from online?
# (Test) importing v-dem datasets online
def getDFfromZip(url):
    """ Return the data frame from a zip file
    Parameters:
        url(string): the url we want to unzip which contains file
    Returns:
        df(pandas DataFrame object): pandas dataframe objects
    """
    response = requests.get(url) # Send a request to download the file
    
    if response.status_code == 200: # Check if the request was successful
        # Read the zip file from the response
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the CSV file within the zip file
            for file in zip_file.namelist():
                if file.endswith(".csv"):
                    # Read the CSV file into a pandas DataFrame
                    df = pd.read_csv(zip_file.open(file))
                    print(df.info())                    
    else: 
        print("Failed to download the dataset.")
        return 0
    
    return df

url = "https://v-dem.net/media/datasets/V-Dem-CY-Core_csv_v13.zip"
vdem_df = getDFfromZip(url)

# %%
vdem_df.shape
# %%
# (Test) data preparation and cleaning
null_columns = vdem_df.columns[vdem_df.isnull().all()] # Find columns containing only null values
# %%
# (Test) 
vdem_21century_df = vdem_df[(vdem_df['year'] >= 2000) & (vdem_df['year'] <= 2022)]
print(vdem_21century_df.head())
print(vdem_21century_df.shape)
#%%
# (Test) import anther dataset and import as a dataframe
from io import StringIO

def getCSVasDF(url):
    # Fetch the CSV data using requests library
    response = requests.get(url)
    csv_data = response.text

    # Convert the CSV data to a pandas DataFrame
    df = pd.read_csv(StringIO(csv_data))
    print(df.head())
    print(df.info())
    print(df.shape)

    return df

url = "https://hdr.undp.org/sites/default/files/2021-22_HDR/HDR21-22_Composite_indices_complete_time_series.csv"
# Display the first few rows of the DataFrame
humanDev_df = getCSVasDF(url)

# %%
column_name = humanDev_df.columns
for col_name in humanDev_df.columns:
    print(col_name)
# %%
# Reshape humanDev_df 


# %%
