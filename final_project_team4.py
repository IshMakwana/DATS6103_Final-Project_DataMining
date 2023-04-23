#%%
"""
DATS 6103 Data Mining - Final Project - Team 4

The dataset 1 used is ... dataset from ... (https://v-dem.net/data/the-v-dem-dataset/ - “V-Dem-CY-Full+Others-v13.csv”).

The project team consists of:
- Daniel Felberg
- Ei Tanaka
- Ishani Makwana
- Tharaka Maddineni 
"""
## About dataset we used: 
"""
The Varieties of Democracy (V-Dem) dataset offers a novel method for envisioning and 
analyzing democracy. It propose a multidimensional and disaggregated dataset that 
captures the complicated nature of democracy as a political system that encompasses 
more than just holding elections. In order to measure these characteristics, the V-Dem 
project separates democracy into five high-level principles: electoral, liberal, participatory, 
deliberative, and and egalitarian, and collects data to measure these principles.
"""
#%%[markdown]
## Data Mining Project-Team4
## Project Title: 
# Team Member: Daniel Felberg, Ei Tanaka, Ishani Makwana, Tharaka Maddineni

# ## Introduction
# Our project is comparing metrics used to determine democracy level  in different countries during the 21st century. The data set was created by V-Dem with 27,555 rows and 808 variables, which measures how democratic countries are using indices, based on numerous factors, including censorship, media bias, political repression, relationship between branches of government, equal rights, and access to democratic processes. The data set also includes “background factors” such as economic, education and other socioeconomic variables.
# The Varieties of Democracy (V-Dem) dataset is a comprehensive tool that allows for a more nuanced understanding of democracy by measuring and analyzing multiple dimensions of democratic practices and institutions. The dataset is based on the idea that democracy is not just about holding free and fair elections, but also about having institutions and practices that promote civil liberties, political equality, citizen participation, and deliberation among different groups.
# To measure these different aspects of democracy, the V-Dem project identifies five high-level principles: electoral democracy, liberal democracy, participatory democracy, deliberative democracy, and egalitarian democracy. These principles are further broken down into more specific indicators and sub-indicators, such as the fairness of electoral processes, the protection of civil liberties, the inclusion of marginalized groups, and the ability of citizens to participate in decision-making processes.
# By using this multidimensional and disaggregated dataset, researchers and policymakers can gain a more nuanced understanding of the strengths and weaknesses of democratic practices and institutions in different countries and regions, and identify areas for improvement.

#%%
# This chunk is for set up modules and libraries
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
import pandas as pd
import geopandas
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import statsmodels.api as sm


#%%
#Import data sets from online
#The code creates a subset of the VDem DataFrame that includes 38 variables of interest. The names of the variables are stored in a list called variables. The code selects the desired columns from the original DataFrame using the .loc[] method and stores the resulting subset in a new DataFrame called vdem_df. 


# def getDFfromZip(url):
#     """ Return the data frame from a csv file in a zip file
#     Parameters:
#         url(str): url of the zip file
#     Returns:
#         df(pandas.DataFrame): data frame of the csv file
#     """
#     response = requests.get(url) # Send a request to download the file
    
#     if response.status_code == 200: # Check if the request was successful
#         # Read the zip file from the response
#         with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
#             # Find the CSV file within the zip file
#             for file in zip_file.namelist():
#                 if file.endswith(".csv"):
#                     # Read the CSV file into a pandas DataFrame
#                     df = pd.read_csv(zip_file.open(file))                    
#                     print(df.shape)      
#                     return df
#             # If the CSV file was not found, return None
#             return None
#     else: 
#         print("Failed to download the dataset.")
#         return None

# url = "https://v-dem.net/media/datasets/V-Dem-CY-FullOthers_csv_v13.zip"
# VDem = getDFfromZip(url)
# VDem.head()
# VDem.shape

#%%
""" Variables of Interest(19):
Infrastracture
- Access_to_clean_fuels_and_technologies_for_cooking_(%_of_population)
- Mobile_cellular_subscriptions_(per_100_people)

Demographic
- Adolescent_fertility_rate_(births_per_1,000_women_ages_15-19)
- Fertility_rate,_total_(births_per_woman)
- Life_expectancy_at_birth,_total_(years)
- Mortality_rate,_under-5_(per_1,000_live_births)
- Net_migration
- Population_growth_(annual_%)

Environment
- CO2_emissions_(metric_tons_per_capita)

Health
- Immunization,_measles_(%_of_children_ages_12-23_months)
- Prevalence_of_HIV,_total_(%_of_population_ages_15-49)

Education
- School_enrollment,_primary_(%_gross)'], dtype=object)

Economy
- Agriculture,_forestry,_and_fishing,_value_added_(%_of_GDP)
- Exports_of_goods_and_services_(%_of_GDP)
- Foreign_direct_investment,_net_inflows_(BoP,_current_US$)
- GDP_(current_US$)', 'GDP_growth_(annual_%)
- GNI_per_capita,_Atlas_method_(current_US$)
- Imports_of_goods_and_services_(%_of_GDP)
"""

# #%% Import data seets from local (World Bank; World Development Indicators) and reformat
# url2 = "https://raw.githubusercontent.com/IshMakwana/DATS6103_Final-Project_DataMining/main/dataset/WorldBank.csv"

# WorldBank_df = pd.read_csv(url2)
# WorldBank_df.head()

# #%%
# # create a dictionary for c_id
# c_id_dict = WorldBank_df.set_index('c_id')['CountryName'].to_dict()
# droped_WorldBank_df = WorldBank_df.drop(['CountryName', 'SeriesName'], axis=1)

# # Reshape the data frame
# new_WB_df = pd.melt(droped_WorldBank_df, id_vars=['c_id', 's_id'], var_name='year', value_name='value')
# new_WB_df['s_id'] = 's_id.' + new_WB_df['s_id'].astype(str)
# new_WB_df = new_WB_df.pivot(index=['c_id', 'year'], columns='s_id', values='value').reset_index()

# # reorder columns
# new_order = ['c_id', 'year', 's_id.1', 's_id.2', 's_id.3', 's_id.4', 's_id.5', 's_id.6', 's_id.7', 's_id.8', 's_id.9', 's_id.10', 's_id.11', 's_id.12', 's_id.13', 's_id.14', 's_id.15', 's_id.16', 's_id.17', 's_id.18', 's_id.19']
# new_WB_df = new_WB_df.reindex(columns=new_order)

# # change column names based on s_id_dict
# new_colnames = {'c_id': 'c_id', 'year': 'year',
#             's_id.1': 'AccessToCleanCooking', 's_id.2': 'AdolescentFertility',
#             's_id.3': 'AgriForestFishValueAdded', 's_id.4': 'CO2Emissions',
#             's_id.5': 'ExportsOfGoodsAndServices', 's_id.6': 'FertilityRate',
#             's_id.7': 'ForeignDirectInvestment', 's_id.8': 'GDP',
#             's_id.9': 'GDPGrowth', 's_id.10': 'GNIPerCapita',
#             's_id.11': 'MeaslesImmunization', 's_id.12': 'ImportsOfGoodsAndServices',
#             's_id.13': 'LifeExpectancy', 's_id.14': 'MobileSubscriptions',
#             's_id.15': 'Under5Mortality', 's_id.16': 'NetMigration',
#             's_id.17': 'PopulationGrowth', 's_id.18': 'HIVPrevalence',
#             's_id.19': 'PrimarySchoolEnrollment'}
# new_WB_df = new_WB_df.rename(columns=new_colnames)

# # remove "yr" from "year" column
# new_WB_df['year'] = new_WB_df['year'].str.replace('yr', '')
# # rename "values of c_id" column" bsed on c_id_dict
# new_WB_df['c_id'] = new_WB_df['c_id'].map(c_id_dict)
# new_WB_df = new_WB_df.rename(columns={'c_id': 'country_name'})

#%% Create a list containing the names of the variables we are interested in
# variables = [
#     "country_name",
#     "country_id",
#     "year",
#     "v2x_polyarchy",
#     "v2x_libdem",
#     "v2x_partipdem",
#     "v2x_delibdem",
#     "v2x_egaldem",
#     "e_regionpol_6C",
# ]
# vdem_df = VDem.loc[:, variables]
# print(vdem_df.shape)
# vdem_df.head()

#%% Create a new democracy index column at right to country_id column from 5 democracy variables
# # Calculate the mean of the five democracy variables for each row
# vdem_df['democracy_index'] = vdem_df[['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 
#                                     'v2x_delibdem', 'v2x_egaldem']].mean(axis=1)

# # Move the new 'democracy_index' column to the right of the 'country_id' column
# columns = list(vdem_df.columns)
# columns.insert(columns.index("year") + 1, columns.pop(columns.index("democracy_index")))
# vdem_df = vdem_df[columns]

# print(vdem_df.shape)
# vdem_df.head()

#%% Create subset containing only 2000s in year column
# # Create a new DataFrame containing only the rows from 2000 onwards
# vdem_worldbank_df = vdem_df.loc[vdem_df["year"] >= 2000]
# # Check the shape of the new DataFrame
# print(vdem_worldbank_df.shape)
# vdem_worldbank_df.head()

#%% Count the number of countries which are included in both vdem_worldbank_df and new_WB_df
# number_of_countries = set(vdem_worldbank_df["country_name"]).intersection(set(new_WB_df["country_name"]))
# number_of_countries

#%% Merge the two dataframes
# vdem_worldbank_df['country_name'] = vdem_worldbank_df['country_name'].replace('United States of America', 'United States')
# vdem_worldbank_df.info()
# new_WB_df.info()
# new_WB_df['year'] = new_WB_df['year'].astype(int)

# new_df = pd.merge(vdem_worldbank_df, new_WB_df, on=["country_name", "year"], how="inner")
# %% Drop the countries where don't have 22 years of data
# counts = new_df['country_name'].value_counts()
# not_22 = counts[counts != 22]
# countries = not_22.index.tolist() # only south sudan has less than 22 years of data
# new_df = new_df[~new_df['country_name'].isin(countries)]

#%% Check and Handle Missing Values
# print(new_df.isnull().sum())

# # Fill null values by meadian among same country name.
# # If there is no value among same country name, fill them by median among same political region.
# # I chose median because the distribution of almost all variables was not normal distribute.
# def fillna(df):
#     # Fill NaN values by median among the same country name
#     df = df.groupby('country_name').apply(lambda x: x.fillna(x.median()))

#     # Fill NaN values by median among the same political region if country median is not available
#     df = df.groupby('e_regionpol_6C').apply(lambda x: x.fillna(x.median()))

#     # Reformat the structure
#     df = df.reset_index(drop=True)

#     return df

#%% Fill missing values
# new_df_2 = fillna(new_df)
# new_df_2.isnull().sum()
# print(new_df_2.shape)
#%% Export the dataset as a new CSV file
#new_df_2.to_csv('dataset/vdem_worldBank.csv', index=False)
#%% EDA

url2 = "https://raw.githubusercontent.com/IshMakwana/DATS6103_Final-Project_DataMining/main/dataset/vdem_worldBank.csv"

vdem_worldBank_df = pd.read_csv(url2)

#%% Basic Information (country time series)
vdem_worldBank_df.head()

#%% Data Types, Data Structure, number of rows and columns, and missing values (country time series)
vdem_worldBank_df.info()

#%% Descriptive Statistics (country time series)
vdem_worldBank_df.describe()

#%% Columns (country time series)
vdem_worldBank_df.columns

#%% Countries (country time series)
vdem_worldBank_df['country_name'].unique() 

#%% Create a new dataframe aggregating the mean of the democracy index for each country
vdem_worldBank_grouped_country = vdem_worldBank_df.groupby('country_name').mean()
vdem_worldBank_grouped_country.reset_index(inplace=True)

#%% Basic Information (grouped by country)
vdem_worldBank_grouped_country.head()

#%% Data Types, number of rows and columns, and missing values (grouped by country)
vdem_worldBank_grouped_country.info()

#%% Data Structure (grouped by country)
vdem_worldBank_grouped_country.shape

#%% Missing Values (grouped by country)
vdem_worldBank_grouped_country.isnull().sum()

#%% Descriptive Statistics (grouped by country)
vdem_worldBank_grouped_country.describe()

#%% Create a new dataframe aggregating the mean of the democracy index for each political region with time series
vdem_worldBank_poli_region = vdem_worldBank_df.groupby(['e_regionpol_6C', 'year']).mean()
vdem_worldBank_poli_region.reset_index(inplace=True)

#%% Basic Information (grouped by political region with time series)
vdem_worldBank_poli_region.head()

#%% Data Types, number of rows and columns, and missing values (grouped by political region with time series)
vdem_worldBank_poli_region.info()

#%% Data Structure (grouped by political region with time series)
vdem_worldBank_poli_region.shape

#%% Descriptive Statistics (grouped by political region with time series)
vdem_worldBank_poli_region.describe()

#%% Create a new dataframe aggregating the mean of the democracy index for each political region
vdem_worldBank_poli_region_grouped = vdem_worldBank_df.groupby('e_regionpol_6C').mean()
vdem_worldBank_poli_region_grouped.reset_index(inplace=True)

#%% Basic Information (grouped by political region)
vdem_worldBank_poli_region_grouped.head()

#%% Data Types, number of rows and columns, and missing values (grouped by political region)
vdem_worldBank_poli_region_grouped.info()

#%% Data Structure (grouped by political region)
vdem_worldBank_poli_region_grouped.shape

#%% Descriptive Statistics (grouped by political region)
vdem_worldBank_poli_region_grouped.describe()

#%% Add the columns for political region dataframe
poli_region_dict = {1: 'Eastern_Europe_and_Central_Asia', 
                    2: 'Latin_America_and_the_Caribbean', 
                    3: 'Middle_East_and_North_Africa', 
                    4: 'Sub-Saharan_Africa', 
                    5: 'Western_Europe_North_America_and_Oceania', 
                    6: 'Asia_and_Pacific'}

vdem_worldBank_df['political_region'] = vdem_worldBank_df['e_regionpol_6C'].map(poli_region_dict)
vdem_worldBank_grouped_country['political_region'] = vdem_worldBank_grouped_country['e_regionpol_6C'].map(poli_region_dict)
vdem_worldBank_poli_region['political_region'] = vdem_worldBank_poli_region['e_regionpol_6C'].map(poli_region_dict)
vdem_worldBank_poli_region_grouped['political_region'] = vdem_worldBank_poli_region_grouped['e_regionpol_6C'].map(poli_region_dict)

#%% Max, Min and Mean of Democracy Index in 2000s (mean of 2000-2021)
max_demo_index = vdem_worldBank_grouped_country['democracy_index'].max()
min_demo_index = vdem_worldBank_grouped_country['democracy_index'].min()
mean_demo_index = vdem_worldBank_grouped_country['democracy_index'].mean()

closest_country = vdem_worldBank_grouped_country.loc[(vdem_worldBank_grouped_country['democracy_index'] - mean_demo_index).abs().argsort()[0], 'country_name']

print(f'Max democracy index: {round(max_demo_index, 2)}, Country: {vdem_worldBank_grouped_country[vdem_worldBank_grouped_country["democracy_index"] == max_demo_index]["country_name"].values[0]}')
print(f'Min democracy index: {round(min_demo_index, 2)}, Country: {vdem_worldBank_grouped_country[vdem_worldBank_grouped_country["democracy_index"] == min_demo_index]["country_name"].values[0]}')
print(f'Mean democracy index: {round(mean_demo_index, 2)}, Country: {closest_country}')

#%% Outliers
num_cols = ['democracy_index','v2x_polyarchy', 'v2x_libdem',          
            'v2x_partipdem', 'v2x_delibdem','v2x_egaldem', 
            'e_regionpol_6C', 'AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions','ExportsOfGoodsAndServices', 
            'FertilityRate','ForeignDirectInvestment','GDP', 'GDPGrowth', 'GNIPerCapita', 
            'MeaslesImmunization','ImportsOfGoodsAndServices', 'LifeExpectancy', 
            'MobileSubscriptions','Under5Mortality', 'NetMigration', 'PopulationGrowth', 
            'HIVPrevalence','PrimarySchoolEnrollment']

# Create a new DataFrame containing only the numeric columns
df_subset = vdem_worldBank_df[num_cols]
# Create a boxplot for each numeric column
sns.boxplot(data = df_subset, orient = "h", palette = "Set2")

#%%[markdown]
# Merging code from merge_data file: 

# %%
#%% Distribution of the variables(with time series)
sns.set_style('darkgrid')

# Loop through columns and create a histogram for each of the numeric columns in the dataframe. 
for col in num_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=vdem_worldBank_df, x=col, kde=True, ax=ax)
    ax.set_title(col.capitalize(), fontsize=14)
    ax.set_xlabel(col.capitalize(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.show()

# A histogram is a graphical representation of the distribution of data, where data values are grouped 
# into bins and the height of each bin represents the number of data values that fall within that bin.

#%% [markdown] Interpreting the results of the distribution plots
# The distributions of the numerical variables in the dataset have varying degrees of skewness. Some of the variables such as foreign direct investment, GDP growth rate, net migration, and population growth rate exhibit normal distribution, while others like adolescent fertility, CO2 emissions, and mortality rate are right-skewed. Democracy index, V-Dem scores, access to clean cooking, and mobile subscriptions are binomially distributed. Understanding the distribution of the variables can help in selecting appropriate statistical methods and interpreting the results accurately.

# Normal Distribution
# - foreign direct investment
# - GDP Growth Rate
# - Net Migration
# - Population Growth Rate
# - Primary School Enrollment
# Right Skewed Distribution
# - Adolescent Fertility
# - Agricultural, Forest, and Fish Value Added (% of GDP)
# - CO2 Emissions (metric tons per capita)
# - Exports of Goods and Services (% of GDP)
# - Fertility Rate
# - GDP (current US$)
# - GNI per capita (current US$)
# - Imports of Goods and Services (% of GDP)
# - Mortality rate, under-5 (per 1,000 live births)
# - HIV Prevalence (% of population ages 15-49)
# Right Skewed Distribution
# - Measles Immunization (% of children ages 12-23 months)
# - Life Expectancy at Birth
# Binomial Distribution
# - Democracy Index
# - V-Dem Polyarchy
# - V-Dem liberal democracy
# - V-Dem participatory democracy
# - V-Dem deliberative democracy
# - V-Dem egalitarian democracy
# - Access to Clean Cooking (% of population)
# - Mobile Subscriptions (per 100 people)

#%%
#%% Boxplot
# The boxplot allows us to compare the distribution of democracy index across regions, as well as 
# identify any potential outliers. It can also help us to identify if there are any significant differences in the median democracy index among different regions.

sns.boxplot(data = vdem_worldBank_df, 
            y = "democracy_index", 
            x = "e_regionpol_6C", dodge=False)
plt.show()

# %% Small multiple time series
# Creating a figure with six subplots, each showing a scatterplot of democracy index versus year for a different geographical region. 
# The data for each subplot is filtered from the original vdem_worldbank_df dataframe based on the value of e_regionpol_6C column.

East_Euro_Central_Asia = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 1]
LatAm_Caribbean = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 2]
Mid_East_North_Africa = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 3]
Sub_Saharan_Africa = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 4]
West_Euro_NA_Oceania = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 5]
Asia_Pacific = vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C'] == 6]

# Create a figure with six subplots
fig, axes = plt.subplots(ncols=3, 
                         nrows=2, 
                         figsize=(12, 10))

# Plot the data on the first subplot
axes[0,0].scatter(East_Euro_Central_Asia['year'], East_Euro_Central_Asia['democracy_index'])
axes[0,0].set_title('East_Euro_Central_Asia')

# Plot the data on the second subplot
axes[0,1].scatter(LatAm_Caribbean['year'], LatAm_Caribbean['democracy_index'])
axes[0,1].set_title('LatAm_Caribbean')

# Plot the data on the third subplot
axes[0,2].scatter(Mid_East_North_Africa['year'], Mid_East_North_Africa['democracy_index'])
axes[0,2].set_title('Mid_East_North_Africa')

# Plot the data on the fourth subplot
axes[1,0].scatter(Sub_Saharan_Africa['year'], Sub_Saharan_Africa['democracy_index'])
axes[1,0].set_title('Sub_Saharan_Africa')

# Plot the data on the fith subplot
axes[1,1].scatter(West_Euro_NA_Oceania['year'], West_Euro_NA_Oceania['democracy_index'])
axes[1,1].set_title('West_Euro_NA_Oceania')

# Plot the data on the sixth subplot
axes[1,2].scatter(Asia_Pacific['year'], Asia_Pacific['democracy_index'])
axes[1,2].set_title('Asia_Pacific')

# show plot
plt.show()


# %% Time series by politico-geographic region
# A line plot of democracy index over time for each politico-geographic region in the dataset 
# It will be useful for visualizing how democracy index has changed over time for different politico-geographic regions.

vdem_worldBank_df['e_regionpol_6C'] = vdem_worldBank_df['e_regionpol_6C'].replace({1: 'Eastern Europe and Central Asia', 2: 'Latin America and the Caribbean', 3: 'Middle East and North Africa', 4: 'Sub-Saharan Africa', 5: 'Western Europe, North America, and Oceania', 6: 'Asia and Pacific'})


sns.set_context("paper")
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'democracy_index', 
            hue =  'e_regionpol_6C', kind = 'line')
plt.show()

# The resulting plot will show how democracy index has changed over time for different politico-geographic regions.
#%% Time series 2
# Creating a line plot of life expectancy over time for each politico-geographic region in the dataset:
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'LifeExpectancy', 
            hue =  'e_regionpol_6C', kind = 'line')
plt.show()

#%% Time series 3
# Creating a line plot of under-5 mortality rate over time for each politico-geographic region in the dataset.
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'Under5Mortality', 
            hue =  'e_regionpol_6C', kind = 'line')
plt.show()

#%% Time series 4
# Creating a line plot of gross national income per capita over time for each politico-geographic region in the dataset.
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'GNIPerCapita', 
            hue =  'e_regionpol_6C', kind = 'line')
plt.show()

#%% Time series 5
# The plot shows how primary school enrollment rates have changed over time for each politico-geographic region in the dataset.
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'PrimarySchoolEnrollment', 
            hue =  'e_regionpol_6C', kind = 'line')
plt.show()

#%%
# 3-d scatter plot

fig = px.scatter_3d(vdem_worldBank_df, 
                    x='year', 
                    y='democracy_index', 
                    z='LifeExpectancy',
              size='democracy_index',
              color='e_regionpol_6C',
              color_continuous_scale='reds',
              opacity=0.7)

fig.update_layout(scene=dict(
                    xaxis_title='Year',
                    yaxis_title='Democracy Index',
                    zaxis_title='Life Expectancy'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))

fig.show()

#%%
# 3-d scatter plot 2

fig = px.scatter_3d(vdem_worldBank_df, 
                    x='year', 
                    y='democracy_index', 
                    z='GNIPerCapita',
              size='democracy_index',
              color='e_regionpol_6C',
              color_continuous_scale='reds',
              opacity=0.7)

fig.update_layout(scene=dict(
                    xaxis_title='Year',
                    yaxis_title='Democracy Index',
                    zaxis_title='GNI Per Capita'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))

fig.show()

#%% Bubble plot animation (attempt #1)
# The  Bubble plot shows the relationship between democracy index and income 
# inequality (measured by the Palma ratio) across different countries in the V-Dem dataset. 
# The size of each bubble represents the country's GDP per capita, and the color represents the country name.

fig, ax = plt.subplots(figsize=(10,6))
sns.set(style="whitegrid")

def update(year):
    # Filter data by year
    data_year = vdem_worldBank_df[vdem_worldBank_df['year'] == year]

    # Updating plot
    ax.clear()
    sns.scatterplot(y = 'democracy_index', 
                    x = 'PrimarySchoolEnrollment', 
                    data = vdem_worldBank_df, 
                    size='GNIPerCapita', hue='country_name', 
                    sizes=(20, 2000), ax=ax)
    
    plt.title('Year: ' + str(year))

# Creating animation
animation = FuncAnimation(fig, 
                          update, 
                          frames=range(2000,2022, 1), 
                          repeat=True)

# saving animation as a GIF
writer = ani.PillowWriter(fps=1,
                        metadata=dict(artist='Me'),
                        bitrate=1800)
animation.save('bubble.gif', writer=writer)

plt.show() # animation won't move here, have to open it in your working directory to see GIF


#%%
# Without size variable of GNIPerCapita

fig, ax = plt.subplots(figsize=(10,6))
sns.set(style="whitegrid")

def update(year):
    # Filter data by year
    data_year = vdem_worldBank_df[vdem_worldBank_df['year'] == year]

    # Updating plot
    ax.clear()
    sns.scatterplot(y = 'democracy_index', 
                    x = 'PrimarySchoolEnrollment', 
                    data = vdem_worldBank_df, 
                    hue = 'country_name', 
                    sizes=(20, 2000), ax=ax)
    
    plt.title('Year: ' + str(year))

# Creating animation
animation = FuncAnimation(fig, 
                          update, 
                          frames=range(2000,2022, 1), 
                          repeat=True)

# saving animation as a GIF
writer = ani.PillowWriter(fps=1,
                        metadata=dict(artist='Me'),
                        bitrate=1800)
animation.save('bubble.gif', writer=writer)

plt.show() # animation won't move here, have to open it in your working directory to see GIF


#%% map of geopolitical regions
# creating a choropleth map of the geopolitical regions using the geopandas library.

# First createing a new dataframe called vdem_worldbank_poli_region_grouped_names with two columns: name and region, which contain the country names and their corresponding geopolitical region codes respectively.
vdem_worldbank_poli_region_grouped_names = pd.DataFrame()

vdem_worldbank_poli_region_grouped_names['name'] = vdem_worldBank_grouped_country['country_name']
vdem_worldbank_poli_region_grouped_names['region'] = vdem_worldBank_grouped_country['e_regionpol_6C']
vdem_worldbank_poli_region_grouped_names.loc['United States', 'country_name'] = 'United States of America'


# Loading the world map using the naturalearth_lowres dataset and filters out Antarctica and countries with a population estimate of zero.
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world = world[(world.pop_est>0) & (world.name!="Antarctica")]
# Creating two new dataframes: country_shapes containing the geometry of each country and its ISO3 code, and country_names containing the name of each country and its ISO3 code.
country_shapes = world[['geometry', 'iso_a3']]
country_names = world[['name', 'iso_a3']]

# Then merging the country_shapes and country_names dataframes on the ISO3 code to get a new dataframe vdem_2000s_grouped_map that contains both the geometry and the name of each country. 
vdem_2000s_grouped_map = country_shapes.merge(country_names, on='iso_a3')
# Merging this dataframe with the vdem_worldbank_poli_region_grouped_names dataframe on the country name to get a new dataframe vdem_2000s_grouped_map2 that contains the geometry, name, and region code of each country.
vdem_2000s_grouped_map2 = vdem_2000s_grouped_map.merge(vdem_worldbank_poli_region_grouped_names, on='name')

ax = vdem_2000s_grouped_map2.plot(column='region')
ax.set_axis_off()

ax.plot()

#%%
# Calculating the Variance Inflation Factor (VIF) for the columns 'demo_mrty(m)_rate' and 'demo_mrty(i)_rate' in the vdem_worldbank_poli_region_grouped dataframe. 

life_expect = vdem_worldBank_df[['Under5Mortality', 'LifeExpectancy']]

vif = pd.DataFrame()

# Creating a new dataframe life_expect that contains only the columns of interest.
vif["VIF Factor"] = [variance_inflation_factor(life_expect.values, i) for i in range(life_expect.shape[1])]
vif["Variable"] = life_expect.columns

vif

#%%[markdown]
# Variables to test against average democracy index: GDP per capita, overall life expectancy, and average education.

#%%[markdown]

# ### Descriptive Statistics
print(vdem_worldBank_poli_region_grouped.columns)

# Select variables of interest
df = vdem_worldBank_df[['democracy_index', 'GNIPerCapita', 
                        'LifeExpectancy', 'PrimarySchoolEnrollment']]

# Print descriptive statistics
print(df.describe())

#%%

#%%[markdown]
# Interpreting the results of the descriptive statistics
# The above descriptive statistics show the count, mean, standard deviation, minimum, maximum, and quartile values of the variables of interest: democracy index, GDP per capita, life expectancy, and average education.
# The demo_index variable has a mean of 0.41 with a standard deviation of 0.24, indicating that the average democracy index across all countries in the dataset is moderate. The minimum value of 0.05 indicates that there are some countries with very low democracy scores, while the maximum value of 0.86 indicates that there are some countries with very high democracy scores.
# The eco_gdp_pc variable has a mean of 14.96 with a standard deviation of 16.37, indicating that there is a wide range of GDP per capita values across the countries in the dataset. The minimum value of 0 indicates that there are some countries with very low levels of economic development, while the maximum value of 84.57 indicates that there are some countries with very high levels of economic development.
# The demo_life_expcy variable has a mean of 68.50 with a standard deviation of 14.23, indicating that the life expectancy across the countries in the dataset is moderate. The minimum value of 0 indicates that there are some countries with very low life expectancy, while the maximum value of 83.53 indicates that there are some countries with very high life expectancy.
# The edu_avg variable has a mean of 5.76 with a standard deviation of 4.39, indicating that the average education level across the countries in the dataset is moderate. The minimum value of 0 indicates that there are some countries with very low levels of education, while the maximum value of 13.24 indicates that there are some countries with very high levels of education.

#%%  Correlation Matrix (Linearity)

cor_mat = vdem_worldBank_df[num_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_mat, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(cor_mat, mask = mask, cmap = cmap, vmax = .3, center=0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5}, 
            fmt = '.1g', annot = True)

plt.title('Correlation matrix')
plt.show()

#%%
#Annova Test 1
from scipy.stats import f_oneway

# Group data by region and calculate mean Democracy Index for each region
dem_index_by_region = vdem_worldBank_df.groupby('e_regionpol_6C')['democracy_index'].mean()

# Perform ANOVA test
f_stat, p_val = f_oneway(*[vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C']==region]['democracy_index'] for region in dem_index_by_region.index])

print(f"ANOVA test result: F-statistic = {f_stat}, p-value = {p_val}")

#%%
#Annova Test 2 

# Group data by region and calculate mean Life Expectancy for each region
life_exp_by_region = vdem_worldBank_df.groupby('e_regionpol_6C')['LifeExpectancy'].mean()

# Perform ANOVA test
f_stat, p_val = f_oneway(*[vdem_worldBank_df[vdem_worldBank_df['e_regionpol_6C']==region]['LifeExpectancy'] for region in life_exp_by_region.index])

print(f"ANOVA test result: F-statistic = {f_stat}, p-value = {p_val}")

#%%
#Annova Test3
# ANOVA test for Child Mortality
grouped_cm = vdem_worldBank_df.groupby('e_regionpol_6C')['Under5Mortality'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_cm)
print("Child Mortality ANOVA test using region:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for GNI Per Capita
grouped_gni = vdem_worldBank_df.groupby('e_regionpol_6C')['GNIPerCapita'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_gni)
print("GNI Per Capita ANOVA test using region:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for Child School Enrollment
grouped_enrollment = vdem_worldBank_df.groupby('e_regionpol_6C')['PrimarySchoolEnrollment'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_enrollment)
print("Child School Enrollment ANOVA test using region:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for Year
grouped_year = vdem_worldBank_df.groupby('e_regionpol_6C')['year'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_year)
print("Year ANOVA test using region:")
print("F value:", f_val)
print("P value:", p_val)

#Annova Test4
# ANOVA test for Child Mortality using country_name
grouped_cm = vdem_worldBank_df.groupby('country_name')['Under5Mortality'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_cm)
print("Child Mortality ANOVA test using country_name:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for GNI Per Capita using country_name
grouped_gni = vdem_worldBank_df.groupby('country_name')['GNIPerCapita'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_gni)
print("GNI Per Capita ANOVA test using country_name:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for Child School Enrollment using country_name
grouped_enrollment = vdem_worldBank_df.groupby('country_name')['PrimarySchoolEnrollment'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_enrollment)
print("Child School Enrollment ANOVA test using country_name:")
print("F value:", f_val)
print("P value:", p_val)

# ANOVA test for Year using country_name
grouped_year = vdem_worldBank_df.groupby('country_name')['year'].apply(list)
f_val, p_val = stats.f_oneway(*grouped_year)
print("Year ANOVA test using country_name:")
print("F value:", f_val)
print("P value:", p_val)

#%%
import pandas as pd
from scipy.stats import f_oneway
# Grouping the vdem_worldBank_df dataframe by country_name
# grouped_df = vdem_worldBank_df.groupby('country_name')
grouped_df = vdem_worldBank_df.groupby('country_name').apply(lambda x: x.fillna(x.median()))
# grouped_df.dropna(inplace=True)
# Creating a dictionary to store the ANOVA results for each variable
anova_dict = {}
target_var = 'democracy_index'
varia_of_interest = ['Under5Mortality', 'GNIPerCapita', 
                     'PrimarySchoolEnrollment', 'year', 
                     'LifeExpectancy', 'e_regionpol_6C']

# Looping through each variable and performing ANOVA
for variable in varia_of_interest:
    # groups = []
    # values = grouped_df[variable]
    # groups.append(values)
    columns_to_extract = [target_var, variable]
    groups = grouped_df[columns_to_extract].apply(list)
    # Performing ANOVA
    f_statistic, p_value = f_oneway(*groups)
    
    # Storing the results in the dictionary
    anova_dict[variable] = {'F-statistic': f_statistic, 'p-value': p_value}

# Displaying the ANOVA results
for variable in anova_dict.keys():
    print(variable)
    print(anova_dict[variable])
    print('\n')

#%% ANOVA test attempt 2

ANOVA_variables = ['e_regionpol_6C','AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions', 'ExportsOfGoodsAndServices', 
            'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 
            'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 
            'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 
            'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

vdem_countries = vdem_worldBank_df['country_name'].unique()

f_stat, p_val = stats.f_oneway(vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==1, 'democracy_index'],
                               vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==2, 'democracy_index'],
                               vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==3, 'democracy_index'],
                               vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==4, 'democracy_index'],
                               vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==5, 'democracy_index'],
                               vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country['e_regionpol_6C']==6, 'democracy_index'])

print('F-statistic: {:.2f}'.format(f_stat))
print('p-value: {:.4f}'.format(p_val))

#%% [markdown] Interpreting the results of the correlation matrix
# The correlation matrix shows the relationship between the democracy index and various factors. 
# A high positive correlation indicates that as the democracy index increases, so do the values of the other factors. 
# In this case, the democracy index is positively correlated with measures of political and economic freedom, 
#   as well as access to education and healthcare. Conversely, the democracy index is negatively correlated with factors 
#   such as adolescent fertility and HIV prevalence. 
# These correlations suggest that democratic societies tend to have better social, economic, and health outcomes.

#%% ttesting

ttest_variables = ['e_regionpol_6C','AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions', 'ExportsOfGoodsAndServices', 
            'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 
            'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 
            'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 
            'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

for x in ttest_variables:
    sample1 = vdem_worldBank_grouped_country['democracy_index'].sample(n=5)
    sample2 = vdem_worldBank_grouped_country[x].sample(n=5)

    t, p, vdem_worldBank_df = sm.stats.ttest_ind(sample1, sample2)

    if p < 0.05:
        print(x)
        print(f't-value: ', t)
        print(f'p-value: ', p)

#%% Bubble Plot
var_independent = ['e_regionpol_6C','AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions', 'ExportsOfGoodsAndServices', 
            'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 
            'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 
            'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 
            'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

# Group by country_name and calculate the mean for each feature
df_subset = vdem_worldBank_grouped_country[var_independent + ['democracy_index']]
df_subset['e_regionpol_6C'] = df_subset['e_regionpol_6C'].astype(int)

features = ['AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions', 
            'ExportsOfGoodsAndServices', 'FertilityRate', 
            'ForeignDirectInvestment','GDP', 'GDPGrowth', 
            'GNIPerCapita', 'MeaslesImmunization', 
            'ImportsOfGoodsAndServices', 'LifeExpectancy', 
            'MobileSubscriptions','Under5Mortality', 'NetMigration', 
            'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

countries = ['United States', 'China', 'India', 'Brazil', 'Japan']

colors = sns.color_palette('bright', n_colors=len(df_subset['e_regionpol_6C'].unique()))

for feature in features:
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='democracy_index', y=feature, size='GDP', sizes=(100, 12000), alpha=0.5, hue='e_regionpol_6C', palette=colors, data=df_subset, legend=False)
    sns.regplot(x='democracy_index', y=feature, data=df_subset, scatter=False, color='black')
    plt.title('Democracy Index vs. ' + feature)

    plt.show()

#%% [markdown] Interpreting the results of the bubble plot
# Use some interesting plot

#%% Time Series Analysis (By Political Region)

plt.figure(figsize=(10, 8))
sns.lineplot(x = 'year', y = 'democracy_index', data = vdem_worldBank_poli_region, 
             hue = 'political_region', legend = True, linewidth = 3)

color_dict = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple', 6: 'brown'}
plt.scatter(x='year', y='democracy_index', data=vdem_worldBank_df, c=vdem_worldBank_df['e_regionpol_6C'].map(color_dict), s=5, alpha=0.2)
plt.xlabel('Year')
plt.ylabel('Democracy Index')
plt.title('Democracy Index over Time')
plt.show()

#%% [markdown] Interpreting the results of the time series analysis

#%% Multicollinearity (VIF test)
X = vdem_worldBank_df[features]

vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

#%% [markdown] Model Building

#%% Regression Tree Model

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X = vdem_worldBank_df[features]
y = vdem_worldBank_df['democracy_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.tree import plot_tree

# Instantiate dt
dt_model = DecisionTreeRegressor(max_depth = 4, 
                                 min_samples_leaf = 0.1, 
                                 random_state = 3)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

#%%
# Decision Tree model (dt_model) evaluation metrics
print("R^2: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(MSE(y_test, y_pred)))
print("MAE: {}".format(MAE(y_test, y_pred)))
print("MSLE: {}".format(MSLE(y_test, y_pred)))
print("MedAE: {}".format(MedAE(y_test, y_pred)))

# Plot the tree graph
plt.figure(figsize = (10, 8))
plot_tree(dt_model, feature_names = X_train.columns, 
          filled = True, rounded = True, fontsize = 12,
          impurity = True, 
          proportion = True, 
          precision = 2)
plt.title('Decision Tree for variables of interest')
plt.show()

#%%

# Plot the feature importances
importances = pd.Series(data=dt_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# Plot the test set with the decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X_test['GDP'], X_test['GDPGrowth'], c=y_test, s=20, cmap='RdYlGn')
plt.title('Test set')
plt.xlabel('GDP')
plt.ylabel('GDP Growth')
plt.show()

#%% [markdown] 
# Interpreting the results of the regression tree model
# The result of the regression tree suggests that the model has moderate predictive power. 
# The R^2 value of 0.54 indicates that the model explains 54% of the variance in the target variable. 
# The MSE value of 0.027 suggests that the average squared difference between the predicted and actual values is relatively low. 
# The MAE value of 0.126 suggests that the average absolute difference between the predicted and actual values is also relatively low.
# The MSLE value of 0.014 indicates that the model's error is distributed logarithmically, 
#     with smaller errors being more common than larger ones. 
# The MedAE value of 0.104 indicates that the median absolute error is relatively low, suggesting that the model is relatively consistent in its predictions.

# Overall, while the model is not perfect, it appears to have some predictive power and is likely to be useful in some applications. 
# However, further analysis and testing may be necessary to fully evaluate its performance.

#%%[DT Model performance]

# Cross validation using sklearn
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_model, 
                         X, y, cv = 5, 
                         scoring='r2')

print("Cross-validation for 5 fold")
print("Cross-validation R^2 scores:", scores)
print("Mean R^2:", scores.mean())

#%% [markdown] 
# Interpreting the results of the regression tree model-cross_validation metrics
# The cross-validation R2 values are: -0.06587845, 0.4301455, 0.32436221, 0.25372059, 0.2647695. 
# A higher R2 value implies that the model fits the data more accurately.

# The average of the R2 values obtained from the folds, as determined by the cross-validation, 
#    is 0.2414238701083093. 
# This result represents how well the model as a whole performed in predicting the democracy index using the provided features.

# An improved model fit is generally indicated by a higher R2 value, 
# but it's also crucial to make sure the model isn't overfitting the data. 
# Cross-validation can give a more precise assessment of the model's performance and aid in evaluating the model's performance on unknown data.


#%%[markdown]
## Ensembling methods
# Random Forest Model

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import plot_tree

# Instantiate rf
rf_model = RandomForestRegressor(n_estimators = 25, 
                                 random_state = 2)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# RF model evaluation metrics

print("R^2: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(MSE(y_test, y_pred)))
print("MAE: {}".format(MAE(y_test, y_pred)))
print("MSLE: {}".format(MSLE(y_test, y_pred)))
print("MedAE: {}".format(MedAE(y_test, y_pred)))

# Plot the feature importances
importances = pd.Series(data = rf_model.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances')
plt.show()

# Plot the test set with the decision boundary
plt.figure(figsize = (10, 8))
plt.scatter(X_test['GDP'], X_test['GDPGrowth'], c = y_test, s = 20, cmap = 'RdYlGn')
plt.title('Test set')
plt.xlabel('GDP')
plt.ylabel('GDP Growth')
plt.show()

#%% [markdown] Interpreting the results of the random forest model
# The random forest model has a high R^2 value of 0.9434, 
#       which means that it can explain 94.34% of the variance in the target variable. 
# The MSE (mean squared error) value is very low at 0.0033, indicating that the model's predictions 
#      are very close to the actual values. 
# The MAE (mean absolute error) value is also low at 0.037, which means that the average difference between the predicted and actual values is small. 
# The MSLE (mean squared logarithmic error) value is low at 0.0018, which indicates that the model is making accurate predictions across the entire range of target values. 
# The MedAE (median absolute error) value is also low at 0.0219, which means that half of the absolute errors are smaller than this value. 

#%%
## Gradient Boosting algorithm (Ensemble) to boost model accuracy and performance

# Instantiate rf
gb_model = GradientBoostingRegressor(n_estimators = 25, 
                                     learning_rate = 0.1, 
                                     random_state = 2)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Gradient Boosting model evaluation metrics

print("R^2: {}".format(r2_score(y_test, y_pred_gb)))
print("MSE: {}".format(MSE(y_test, y_pred_gb)))
print("MAE: {}".format(MAE(y_test, y_pred_gb)))
print("MSLE: {}".format(MSLE(y_test, y_pred_gb)))
print("MedAE: {}".format(MedAE(y_test, y_pred_gb)))

# Plot the feature importances
importances = pd.Series(data = gb_model.feature_importances_, 
                        index = X.columns)

importances_sorted = importances.sort_values()
importances_sorted.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances (GB)')
plt.show()

# Plot the test set with the decision boundary
plt.figure(figsize = (10, 8))
plt.scatter(X_test['GDP'], X_test['GDPGrowth'], c = y_test, s = 20, cmap = 'RdYlGn')
plt.title('Test set')
plt.xlabel('GDP')
plt.ylabel('GDP Growth')
plt.show()

#%% [markdown] Interpreting the results of the Gradient Boosting model:

# The gradient boosting model has a R^2 value of 0.7118, 
#       which means that it can explain 71.18% of the variance in the target variable. 
# The MSE (mean squared error) value is very low at 0.017, indicating that the model's predictions 
#      are close to the actual values. 
# The MAE (mean absolute error) value is also low at 0.107, which means that the average difference between the predicted and actual values is small. 
# The MSLE (mean squared logarithmic error) value is low at 0.0092, which indicates that the model is making accurate predictions across the entire range of target values. 
# The MedAE (median absolute error) value is also low at 0.0926, which means that half of the absolute errors are smaller than this value. 

#%%[markdown]
# Overall, these results suggest that the random forest model is a good fit over regression and gradient boosting models 
#     for the data and is making accurate predictions with R^2 (R-squared) value of 94.3%.

#%% [markdown] Compare the results of the regression tree and random forest models
# To compare the performance of the regression tree, gradient boosting and random forest models, 
# we can look at the evaluation metrics such as: 
#   R-squared (R^2), 
#   mean squared error (MSE), 
#   mean absolute error (MAE), 
#   mean squared logarithmic error (MSLE), and 
#   median absolute error (MedAE). 
# We can also compare the feature importances of the models.
# From the evaluation metrics, we can see that the random forest model outperforms the regression tree model and gradient boosting model on all metrics. 
# The R^2 value of the random forest model is 0.9434, which is much higher than the R^2 value of the regression tree model (0.5426). 
# The MSE, MAE, MSLE, and MedAE values of the random forest model are also lower than those of the regression tree and gradient boosting models, 
#       indicating that the random forest model is making more accurate predictions.
# We can also compare the feature importances of the two models: 
#     The feature importances of the random forest model are generally higher than those of the regression tree & gradient boosting models, 
#       indicating that the random forest model is able to better capture the relationships between the features and the target variable.
# Overall, the random forest model appears to be a better choice for this dataset as it outperforms the regression tree and gradient boosting model on all evaluation metrics and has higher feature importances.

#%% Refining the random forest model

# Hyperparameter tuning
# GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [25, 50, 75],
    'max_depth': [2, 4, 6, None],
    'min_samples_leaf': [0.1, 0.2, 0.3]
}

# Instantiate rf
rf = RandomForestRegressor(random_state=2)

# Instantiate grid_rf
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Fit GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters: ", grid_search.best_params_)

# Use best parameters to fit the model
rf_best = RandomForestRegressor(n_estimators=grid_search.best_params_            
                                ['n_estimators'], 
                                max_depth=grid_search.best_params_['max_depth'], 
                                min_samples_leaf=grid_search.best_params_['min_samples_leaf'], 
                                random_state=2)
rf_best.fit(X_train, y_train)
y_pred = rf_best.predict(X_test)

# Evaluate the model
print("R^2: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(MSE(y_test, y_pred)))
print("MAE: {}".format(MAE(y_test, y_pred)))
print("MSLE: {}".format(MSLE(y_test, y_pred)))
print("MedAE: {}".format(MedAE(y_test, y_pred)))

# Plot Tree Graph
plt.figure(figsize=(20, 10))
plot_tree(rf_best.estimators_[0], feature_names=X_train.columns, filled=True, rounded=True)
plt.show()


# Plot the feature importances
importances = pd.Series(data=rf_best.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

# Plot the test set with the decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X_test['GDP'], X_test['GDPGrowth'], c=y_test, s=20, cmap='RdYlGn')
plt.title('Test set')
plt.xlabel('GDP')
plt.ylabel('GDP Growth')
plt.show()

#%% [markdown] Interpreting the results of the refined random forest model
# After refining the random forest model using GridSearchCV, 
# the R^2 value decreased to 0.5205, indicating that the model's ability to explain the variance in the target variable has decreased. 
# The MSE value increased to 0.0283, indicating that the average squared difference between the predicted and actual values has increased. 
# The MAE value increased to 0.1358, indicating that the average absolute difference between the predicted and actual values has also increased.
# The MSLE value of 0.0149 indicates that the model's error is distributed logarithmically, with smaller errors being more common than larger ones. 
# The MedAE value of 0.1146 indicates that the median absolute error is higher than the previous model.

# Overall, while the refined random forest model has a lower performance than the previous one, it is still making relatively accurate predictions. 
# Further analysis and testing may be necessary to fully evaluate its performance.

#%% [markdown] Results
# - The random forest model has a high R^2 value of 0.9434, which means that it can explain 94.34% of the variance in the target variable.
# - The MSE (mean squared error) value is very low at 0.0033, indicating that the model's predictions are very close to the actual values.
# - The MAE (mean absolute error) value is also low at 0.037, which means that the average difference between the predicted and actual values is small.
# - The MSLE (mean squared logarithmic error) value is low at 0.0018, which indicates that the model is making accurate predictions across the entire range of target values.
# - The MedAE (median absolute error) value is also low at 0.0219, which means that half of the absolute errors are smaller than this value.
# - Overall, these results suggest that the random forest model is a good fit for the data and is making accurate predictions.

# %%
