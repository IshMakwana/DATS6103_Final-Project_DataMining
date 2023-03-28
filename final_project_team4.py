"""
DATS 6103 - Final Project - Team 4

The dataset 1 used is ... dataset from ... (https://).

The project team consists of:
- Daniel Felberg
- Ei Tanaka
- Ishani Makwana
- Tharaka Maddineni 
"""
#%%[markdown]
# # Title
# Team Member: 1, 2, 3, 4

# ## Introduction

#%%
# This chunk is for set up mmodules and libraries
# Set up library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import requests
import io
import zipfile
from io import StringIO
#%%
# Import data sets from online
def getDFfromZip(url):
    """ Return the data frame from a csv file in a zip file
    Parameters:
        url(str): url of the zip file
    Returns:
        df(pandas.DataFrame): data frame of the csv file
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
                    print(df.shape)      
                    return df
            # If the CSV file was not found, return None
            return None
    else: 
        print("Failed to download the dataset.")
        return None

url = "https://v-dem.net/media/datasets/V-Dem-CY-FullOthers_csv_v13.zip"
vdem_df = getDFfromZip(url)
vdem_df.head()
vdem_df.shape

#%%[markdown]
# ## Data Cleaning and Prepareation

#%%
col_name = 'e_peaveduc'
print(vdem_df.loc[vdem_df['country_name'] == 'Mexico', col_name])

col_name = 'e_area'
print(vdem_df.loc[vdem_df['country_name'] == 'Mexico', col_name])

col_name = 'e_pop'
print(vdem_df.loc[vdem_df['country_name'] == 'Mexico', col_name])

col_name = 'e_radio_n'
print(vdem_df.loc[vdem_df['country_name'] == 'Mexico', col_name])

col_name = 'e_pefeliex'
print(vdem_df.loc[vdem_df['country_name'] == 'Mexico', col_name])
#%%[markdown]
# ## EDA

#%%

#%%[markdown]
# ### Basic EDA

#%%

#%%[markdown]
# Interpreting the results of the basic EDA

#%%[markdown]
# ### Descriptive Statistics

#%%

#%%[markdown]
# Interpreting the results of the descriptive statistics

#%%[markdown]
# ### Hypothesis Testing(Correlation, Regression, etc.)

#%%

#%%[markdown]
# Interpreting the results of the hypothesis testing

#%%[markdown]
# ### Correlation Analysis

#%%

#%%[markdown]
# Interpreting the results of the correlation analysis

#%%[markdown]
# ## Model Building

#%%

#%%[markdown]
# ## Result

#%%[markdown]
# ## Discussion (Limitations, Future Work, etc.)
#%%[markdown]
# ## Conclusion

#%%[markdown]
# ## References