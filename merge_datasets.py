#%% This chunk is for set up modules and libraries
# Set up library
# Import all required list of libraries here. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import io
import zipfile
from io import StringIO
from scipy.stats import shapiro
import gapminder
import matplotlib.animation as ani
from matplotlib.animation import FuncAnimation
import plotly.express as px
import random
import plotly.graph_objs as go

#%%
## About dataset we used: 
"""
The Varieties of Democracy (V-Dem) dataset offers a novel method for envisioning and 
analyzing democracy. It propose a multidimensional and disaggregated dataset that 
captures the complicated nature of democracy as a political system that encompasses 
more than just holding elections. In order to measure these characteristics, the V-Dem 
project separates democracy into five high-level principles: electoral, liberal, participatory, 
deliberative, and and egalitarian, and collects data to measure these principles.
"""

#%% [markdown]
# ## Data Preparation(merging data set) & Cleaning
#%% Import data sets from online (V-Dem)

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
VDem = getDFfromZip(url)
VDem.head()
VDem.shape

#%% Import data seets from local (World Bank; World Development Indicators) and reformat
url2 = "https://raw.githubusercontent.com/IshMakwana/DATS6103_Final-Project_DataMining/main/dataset/WorldBank.csv"

WorldBank_df = pd.read_csv(url2)
WorldBank_df.head()

#%%
# create a dictionary for c_id
c_id_dict = WorldBank_df.set_index('c_id')['CountryName'].to_dict()
droped_WorldBank_df = WorldBank_df.drop(['CountryName', 'SeriesName'], axis=1)

# Reshape the data frame
new_WB_df = pd.melt(droped_WorldBank_df, id_vars=['c_id', 's_id'], var_name='year', value_name='value')
new_WB_df['s_id'] = 's_id.' + new_WB_df['s_id'].astype(str)
new_WB_df = new_WB_df.pivot(index=['c_id', 'year'], columns='s_id', values='value').reset_index()

# reorder columns
new_order = ['c_id', 'year', 's_id.1', 's_id.2', 's_id.3', 's_id.4', 's_id.5', 's_id.6', 's_id.7', 's_id.8', 's_id.9', 's_id.10', 's_id.11', 's_id.12', 's_id.13', 's_id.14', 's_id.15', 's_id.16', 's_id.17', 's_id.18', 's_id.19']
new_WB_df = new_WB_df.reindex(columns=new_order)

# change column names based on s_id_dict
new_colnames = {'c_id': 'c_id', 'year': 'year',
            's_id.1': 'AccessToCleanCooking', 's_id.2': 'AdolescentFertility',
            's_id.3': 'AgriForestFishValueAdded', 's_id.4': 'CO2Emissions',
            's_id.5': 'ExportsOfGoodsAndServices', 's_id.6': 'FertilityRate',
            's_id.7': 'ForeignDirectInvestment', 's_id.8': 'GDP',
            's_id.9': 'GDPGrowth', 's_id.10': 'GNIPerCapita',
            's_id.11': 'MeaslesImmunization', 's_id.12': 'ImportsOfGoodsAndServices',
            's_id.13': 'LifeExpectancy', 's_id.14': 'MobileSubscriptions',
            's_id.15': 'Under5Mortality', 's_id.16': 'NetMigration',
            's_id.17': 'PopulationGrowth', 's_id.18': 'HIVPrevalence',
            's_id.19': 'PrimarySchoolEnrollment'}

new_WB_df = new_WB_df.rename(columns=new_colnames)

# remove "yr" from "year" column
new_WB_df['year'] = new_WB_df['year'].str.replace('yr', '')
# rename "values of c_id" column" bsed on c_id_dict
new_WB_df['c_id'] = new_WB_df['c_id'].map(c_id_dict)
new_WB_df = new_WB_df.rename(columns={'c_id': 'country_name'})

#%% Create a list containing the names of the variables we are interested in
variables = [
    "country_name",
    "country_id",
    "year",
    "v2x_polyarchy",
    "v2x_libdem",
    "v2x_partipdem",
    "v2x_delibdem",
    "v2x_egaldem",
    "e_regionpol_6C",
]
vdem_df = VDem.loc[:, variables]
print(vdem_df.shape)
vdem_df.head()

#%% Create a new democracy index column at right to country_id column from 5 democracy variables
# Calculate the mean of the five democracy variables for each row
vdem_df['democracy_index'] = vdem_df[['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 
                                    'v2x_delibdem', 'v2x_egaldem']].mean(axis=1)

# Move the new 'democracy_index' column to the right of the 'country_id' column
columns = list(vdem_df.columns)
columns.insert(columns.index("year") + 1, columns.pop(columns.index("democracy_index")))
vdem_df = vdem_df[columns]

print(vdem_df.shape)
vdem_df.head()
#%% Create subset containing only 2000s in year column
# Create a new DataFrame containing only the rows from 2000 onwards
vdem_2000s_df = vdem_df.loc[vdem_df["year"] >= 2000]
# Check the shape of the new DataFrame
print(vdem_2000s_df.shape)
vdem_2000s_df.head()

#%% Count the number of countries which are included in both vdem_2000s_df and new_WB_df
number_of_countries = set(vdem_2000s_df["country_name"]).intersection(set(new_WB_df["country_name"]))
number_of_countries

#%% Merge the two dataframes
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('United States of America', 'United States')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('United States of America', 'United States')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Democratic Republic of the Congo', 'Congo, Dem. Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Republic of the Congo', 'Congo, Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Syria', 'Syrian Arab Republic')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Burma/Myanmar', 'Myanmar')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('The Gambia', 'Gambia, The')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Hong Kong', 'Hong Kong SAR, China')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Egypt', 'Egypt, Arab Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Venezuela', 'Venezuela, RB')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Russia', 'Russian Federation')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('North Korea', "Korea, Dem. People's Rep.")
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('South Korea', 'Korea, Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Iran', 'Iran, Islamic Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Laos', 'Lao PDR')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Yemen', 'Yemen, Rep.')
vdem_2000s_df['country_name'] = vdem_2000s_df['country_name'].replace('Turkey', 'Turkiye')


vdem_2000s_df.info()
new_WB_df.info()
new_WB_df['year'] = new_WB_df['year'].astype(int)

new_df = pd.merge(vdem_2000s_df, new_WB_df, on=["country_name", "year"], how="inner")

# %% Drop the countries where don't have 22 years of data
counts = new_df['country_name'].value_counts()
not_22 = counts[counts != 22]
countries = not_22.index.tolist() # only south sudan has less than 22 years of data
new_df = new_df[~new_df['country_name'].isin(countries)]

#%% Check and Handle Missing Values
# print(new_df.isnull().sum())

# Fill null values by meadian among same country name.
# If there is no value among same country name, fill them by median among same political region.
# I chose median because the distribution of almost all variables was not normal distribute.
def fillna(df):
    # Fill NaN values by median among the same country name
    df = df.groupby('country_name').apply(lambda x: x.fillna(x.median()))

    # Fill NaN values by median among the same political region if country median is not available
    df = df.groupby('e_regionpol_6C').apply(lambda x: x.fillna(x.median()))

    # Reformat the structure
    df = df.reset_index(drop=True)

    return df

#%% Fill missing values
new_df_2 = fillna(new_df)
new_df_2.isnull().sum()
print(new_df_2.shape)
#%% Export the dataset as a new CSV file
new_df_2.to_csv('dataset/vdem_worldBank.csv', index=False)