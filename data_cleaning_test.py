"""
Data Cleaning and Preparation
"""
#%%
"""
This chunk is for set up libraries
"""
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
from io import StringIO

#%%
"""
In this chunk, we will import data sets from online and import as a dataframe
"""

# Import data sets from online
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
                    print(df.shape)      
                    return df
            # If the CSV file was not found, return None
            return None
    else: 
        print("Failed to download the dataset.")
        return None

url = "https://v-dem.net/media/datasets/V-Dem-CY-Core_csv_v13.zip"
vdem_df = getDFfromZip(url)
vdem_df.head()

# %%
# (Test) data preparation and cleaning for vdem_df
null_columns = vdem_df.columns[vdem_df.isnull().all()] # Find columns containing only null values
# %%
# (Test) Create a new dataframe with only the data from the 21st century
vdem_21century_df = vdem_df[(vdem_df['year'] >= 2000) & (vdem_df['year'] <= 2022)]
print(vdem_21century_df.shape)
vdem_21century_df.head()

#%%
"""
This chunk is for importing data sets from online and import as a dataframe named humanDev_df
"""

# (Test) import anther dataset and import as a dataframe
def getCSVasDF(url):
    # Fetch the CSV data using requests library
    response = requests.get(url)
    csv_data = response.text

    # Convert the CSV data to a pandas DataFrame
    df = pd.read_csv(StringIO(csv_data))
    print(df.shape)

    return df

url = "https://hdr.undp.org/sites/default/files/2021-22_HDR/HDR21-22_Composite_indices_complete_time_series.csv"
# Display the first few rows of the DataFrame
humanDev_df = getCSVasDF(url)
humanDev_df.head()
# %%
"""
This chunk is for data preparation and cleaning for humanDev_df
"""

# (Test) data preparation and cleaning for humanDev_df

# Drop unnecessary columns
columns_to_drop = ['iso3', 'hdicode', 'region', 'hdi_rank_2021']
humanDev_wide_df = humanDev_df.drop(columns_to_drop, axis=1)

# humanDev_wide_df.shape
# humanDev_wide_df.head()
# humanDev_wide_df.tail()

# Drop unnecessary time Series
years_to_remove = [str(year) for year in range(1990, 2000)]
cols_to_remove = [col for col in humanDev_wide_df.columns if any(year in col for year in years_to_remove)]
humanDev_wide_df = humanDev_wide_df.drop(cols_to_remove, axis=1)

# humanDev_wide_df.shape
# humanDev_wide_df.head()
# humanDev_wide_df.tail()

# Reshape humanDev_wide_df into long_df 
df_long = pd.melt(humanDev_wide_df, id_vars='country', var_name='measures', value_name='value')

# Sort the DataFrame by country and measures
df_long_sorted = df_long.sort_values(by=['country', 'measures'])
df_long_sorted.reset_index(drop=True, inplace=True)

# df_long_sorted.shape
# df_long_sorted.head()
# df_long_sorted.tail()

# Split the "measures" column into two columns ("prefix" and "year")
df_long_sorted[['measure_id', 'year']] = df_long_sorted['measures'].str.rsplit('_', n=1, expand=True)

# df_long_sorted.shape
# df_long_sorted.head()
# df_long_sorted.tail()

# Drop the "measures" column
humanDev_long_df = df_long_sorted.drop('measures', axis=1)

# humanDev_long_df.shape
# humanDev_long_df.head()
# humanDev_long_df.tail()

# Change the data type of the "year" column to integer
humanDev_long_df['year'] = humanDev_long_df['year'].astype(int)

# Change the order of the columns
humanDev_long_df = humanDev_long_df[['country', 'year', 'measure_id', 'value']]

# humanDev_long_df.shape
# humanDev_long_df.head()
# humanDev_long_df.tail()

# Resahape the DataFrame to a wide format
humanDev_21century_df = humanDev_long_df.pivot_table(index=['country', 'year'], columns='measure_id', values='value')

humanDev_21century_df

# Print the sum of null values in each column
print(humanDev_long_df.isnull().sum())

#%% (Test) Generalize each dataset to Generalize data for each country based on data for each country for each year
"""
vdem_21century_df: Degree of Democracy in Each Country in the 21st Century

humanDev_21centry_df: Degree of human resources growth in each country in the 21st century

we can add more datasets to the list
"""

#%%
# (Test) merge vdem_21century_df and humanDev_final on country and yearvdem_21century_df
new_df = pd.merge(vdem_21century_df, humanDev_21century_df, how='inner', left_on=['country_name', 'year'], right_on=['country', 'year'])
# %%
print(new_df.shape)
new_df.head()
# %%
new_df.tail()