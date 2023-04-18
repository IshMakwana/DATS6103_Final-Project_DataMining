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

#%%
# Import data sets from online
# The code creates a subset of the VDem DataFrame that includes 38 variables of interest. The names of the variables are stored in a list called variables. The code selects the desired columns from the original DataFrame using the .loc[] method and stores the resulting subset in a new DataFrame called vdem_df. 

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

#%%[markdown]
# ## Data Pre-Processing

"""
Variables of interest (38 in totall): original variables in the dataset 

- country_name(str)
- country_id(int)
- year(int)

<independent variables>
- v2x_polyarchy (int): Electorical democracy index
- v2x_libdem (int): Liberal democracy index
- v2x_partipdem (int): Participatory democracy index
- v2x_delibdem (int): Deliberative democracy index
- v2x_egaldem (int): Egalitarian democracy index

< dependent variables>
(Education: 2 variables)
- e_peaveduc (float): The average years of education among citizens older than 15
- e_peedgini (float): Education inequality (Gini coefficient)

(Geography: 4 variables)
- e_area (float): Area of the country (km2)
- e_regiongeo (int): Region (geographic)
- e_regionpol (int): Region (politico-geographic)
- e_regionpol_6C (int): Region (politico-geographic 6-category)

(Economic: 6 variables)
- e_cow_exports (float): the total value of a country's exports
- e_cow_imports (float): the total value of a country's imports
- e_gdp (float): GDP
- e_gdppc (float): GDP per capita
- e_miinflat (float): Annual inflation rate
- e_pop (float): Population

(Natural Resources Wealth: 3 variables)
- e_total_fuel_income_pc (float): the real value of a country's petroleum, coal, and natural gas production
- e_total_oil_income_pc (float): the real value of a country's petroleum production
- e_total_resources_income_pc (float): the real value of a country's petroleum, coal, natural gas, and mineral production

(Infrastructure: 1 variable)
- e_radio_n (int): the number of radio sets

(Demography: 9 variables)
- e_miferrat (float): the fertility rate
- e_mipopula (int): the total population (in thousands
- e_miurbani (float): the urbanization rate
- e_miurbpop (int): the urban population (in thousands)
- e_pefeliex (float): the life expectancy at birth among women
- e_peinfmor (float): the infant mortality rate
- e_pelifeex (float): the life expectancy
- e_pematmor (float): the maternal mortality rate
- e_wb_pop (int): the total population (in thousands)

(Conflict: 5 variables)
- e_civil_war (boolean): Was there a civil war?
- e_miinteco (boolean): Did the country participate in an international armed conflict?
- e_miinterc (boolean): Did the country experience an internal armed conflict?
- e_pt_coup (int): Number of successful coup attempts in a year
- e_pt_coup_attempts (int): Number of coups attempts in a year
"""

#%%
# Step 1: Create subset containing only the columns of interest (38 variables)
# Create a list containing the names of the variables you want to select

variables = [
    "country_name",
    "country_id",
    "year",
    "v2x_polyarchy",
    "v2x_libdem",
    "v2x_partipdem",
    "v2x_delibdem",
    "v2x_egaldem",
    "e_peaveduc",
    "e_peedgini",
    "e_area",
    "e_regiongeo",
    "e_regionpol",
    "e_regionpol_6C",
    "e_cow_exports",
    "e_cow_imports",
    "e_gdp",
    "e_gdppc",
    "e_miinflat",
    "e_pop",
    "e_total_fuel_income_pc",
    "e_total_oil_income_pc",
    "e_total_resources_income_pc",
    "e_radio_n",
    "e_miferrat",
    "e_mipopula",
    "e_miurbani",
    "e_miurbpop",
    "e_pefeliex",
    "e_peinfmor",
    "e_pelifeex",
    "e_pematmor",
    "e_wb_pop",
    "e_civil_war",
    "e_miinteco",
    "e_miinterc",
    "e_pt_coup",
    "e_pt_coup_attempts",
    "v2smpardom",

]

# Select the desired columns from the DataFrame
vdem_df = VDem.loc[:, variables]

# Check the shape of the new DataFrame
print(vdem_df.shape)

vdem_df.head()

#%%
# Step 2: Create a new democracy index column at right to country_id column from 5 democracy variables
# Calculate the mean of the five democracy variables for each row

vdem_df['democracy_index'] = vdem_df[['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 
                                      'v2x_delibdem', 'v2x_egaldem']].mean(axis=1)

# Move the new 'democracy_index' column to the right of the 'country_id' column
columns = list(vdem_df.columns)
columns.insert(columns.index("year") + 1, columns.pop(columns.index("democracy_index")))
vdem_df = vdem_df[columns]

# Check the shape of the new DataFrame
print(vdem_df.shape)
vdem_df.head()

#%% 
# Step 3: Create subset containing only 2000s in year column
# Create a new DataFrame containing only the rows from 2000 onwards

vdem_2000s_df = vdem_df.loc[vdem_df["year"] >= 2000]

# Check the shape of the new DataFrame
print(vdem_2000s_df.shape)
vdem_2000s_df.head()

#%%
# Step 4: Combine the datasets by country (Combine multiple years into one and remove year column)

# Set 'country_id' and 'country_name' as a multi-level index
vdem_2000s_df_indexed = vdem_2000s_df.set_index(['country_name', 'country_id'])

# Setting the name of the dataframe as same as its variable name
vdem_2000s_df_indexed.Name = "vdem_2000s_df_indexed"

# Group by 'country_id' and 'country_name', and aggregate the mean
vdem_2000s_grouped_df = vdem_2000s_df_indexed.groupby(['country_name', 'country_id']).agg("mean")

# Reset the index, so 'country_name' becomes a column again
vdem_2000s_grouped_df = vdem_2000s_grouped_df.reset_index()

# Remove the 'year' column
vdem_2000s_grouped_df = vdem_2000s_grouped_df.drop(columns=["year"])

# Display the combined DataFrame
print(vdem_2000s_grouped_df.shape)
vdem_2000s_grouped_df.head()

#%%[markdown]
### Dataframe variables we created so far(Step 1-4):

"""
VDem: original data set
# 
vdem_df: subset containing only the columns of interest (38 variables) + democracy index
#  
vdem_2000s_df: subset containing only 2000s in year column
# 
vdem_2000s_grouped_df: combine the datasets by country (Combine multiple years into one and remove year column)
"""

#%%
# Step 5: Test 1 (If anything goes wrong, just go back and check Step 1)

#%%[markdown]
# We are defining a set of variables of interest for your analysis. You have defined the independent variables which include various measures of democracy such as the electoral, liberal, participatory, deliberative, and egalitarian democracy indices. The dependent variables include measures of education, geography, economy, natural resources wealth, infrastructure, demography, and conflict.

'''
# # Variables of interest
country_name: str
country_id: int
year: int

# Independent variables
v2x_polyarchy: int # Electoral democracy index
v2x_libdem: int # Liberal democracy index
v2x_partipdem: int # Participatory democracy index
v2x_delibdem: int # Deliberative democracy index
v2x_egaldem: int # Egalitarian democracy index

# Dependent variables - Education
edu_avg_years: float # The average years of education among citizens older than 15
edu_inequality: float # Education inequality (Gini coefficient)

# Dependent variables - Geography
geo_area: float # Area of the country (km2)
geo_region_geographic: int # Region (geographic)
geo_region_politico_geographic: int # Region (politico-geographic)
geo_region_politico_geographic_6C: int # Region (politico-geographic 6-category)

# Dependent variables - Economic
econ_exports: float # the total value of a country's exports
econ_imports: float # the total value of a country's imports
econ_gdp: float # GDP
econ_gdp_per_capita: float # GDP per capita
econ_annual_inflation_rate: float # Annual inflation rate
econ_population: float # Population

# Dependent variables - Natural Resources Wealth
natres_total_fuel_income_per_capita: float # the real value of a country's petroleum, coal, and natural gas production
natres_total_oil_income_per_capita: float # the real value of a country's petroleum production
natres_total_resources_income_per_capita: float # the real value of a country's petroleum, coal, natural gas, and mineral production

# Dependent variables - Infrastructure
infra_num_radio_sets: int # the number of radio sets

# Dependent variables - Demography
demo_fertility_rate: float # the fertility rate
demo_total_population: int # the total population (in thousands)
demo_urbanization_rate: float # the urbanization rate
demo_urban_population: int # the urban population (in thousands)
demo_life_expectancy_women: float # the life expectancy at birth among women
demo_infant_mortality_rate: float # the infant mortality rate
demo_life_expectancy: float # the life expectancy
demo_maternal_mortality_rate: float # the maternal mortality rate
demo_total_population_wb: int # the total population (in thousands)

# Dependent variables - Conflict
conflict_civil_war: bool # Was there a civil war?
conflict_international_armed_conflict: bool # Did the country participate in an international armed conflict?
conflict_internal_armed_conflict: bool # Did the country experience an internal armed conflict?
conflict_successful_coup_attempts: int # Number of successful coup attempts in a year
conflict_coup_attempts: int # Number of coups attempts in a year
'''
#%%
# Step 6:  changing variable names
variable_names = ['country_name', 'country_id', 'demo_index', 'elec_demo_idx', 'lib_demo_idx', 
                  'parti_demo_idx', 'deli_demo_idx', 'ega_demo_idx', 'edu_avg', 'edu_ineql', 
                  'geo_area', 'geo_rgn_geo', 'geo_reg_polc_g', 'geo_reg_polc_g6c', 'eco_exports', 
                  'eco_imports', 'eco_gdp', 'eco_gdp_pc', 'eco_a_ifl_rate', 'eco_popln', 
                  'n_ttl_fuel_income_pc', 'n_ttl_oil_income_pc', 'n_ttl_res_income_pc', 
                  'infra_radio(n)_sets', 'demo_frtly_rate', 'demo_ttl_popln', 'demo_urbzn_rate', 'demo_urbn_popln', 
                  'demo_lf_expcy(w)', 'demo_mrty(i)_rate', 'demo_life_expcy', 'demo_mrty(m)_rate', 'demo_ttl_popln_wb', 'c_civil_war', 
                  'c_intnl_arm_c', 'c_intl_arm_c', 'c_suc_coup_attp', 'c_coup_attp', 'party_dissem']

vdem_2000s_grouped_df.columns = variable_names

#%%[markdown]
## Step 7: Data Cleaning(drop null values, replacing the null values,  drop duplicates rows, etc.)

#%%[markdown]
### 7.1 Check - Identifying and replacing null values
def mean_median_imputation(column : pd.Series):
    """
    This code uses the shapiro() function from the scipy.stats module to perform the Shapiro-Wilk test, 
    which is a statistical test used to determine if a set of data is normally distributed.
    If normally distributed -> mean imputation
    If data is skewed -> median imputation

    Keyword arguments:
    column : V-dem dataframe column of type - Pandas Series

    Retuns:
    mean/median : a floating number (mean/median) of the column
    """

    stat, p = shapiro(column)
    alpha = 0.05
    
    if p >= alpha:
        # print(f'The distribution of {column} is normal (p={p})')
        return column.mean()
    else:
        # print(f'The distribution of {column} is skewed (p={p})')
        return column.median()


def fill_na(df):
    """
    This function replace the null values with mean or median value,
    within in the dataframe and returns the cleaned dataframe

    Keyword arguments:
    df : V-dem dataframe 

    Returns:
    cleaned_df : Clean dataframe
    """
    # fill_null_0 = lambda col: col.fillna(0) if col.dtype in ['float64', 'int64'] else col.fillna(method='ffill')
    # df = df.apply(fill_null_0)

    # Replacing null values with appropriate value (either mean or median)
    fill_null_imputation = lambda col: col.fillna(mean_median_imputation(column = col)) if col.dtype in ['float64', 'int64'] else col.fillna(method='ffill')
    cleaned_df = df.apply(fill_null_imputation)

    return cleaned_df


# Cleaning data for vdem_2000s_df dataframe
print(f"Count of null records in the dataframe before cleaning: {vdem_2000s_df.isnull().sum()}")
vdem_2000s_df = fill_na(df = vdem_2000s_df)
print(f"Count of null records in the dataframe after cleaning: {vdem_2000s_df.isnull().sum()}")
    

# Cleaning data for vdem_2000s_grouped_df dataframe
print(f"Count of null records in the dataframe before cleaning: {vdem_2000s_grouped_df.isnull().sum()}")
vdem_2000s_grouped_df = fill_na(df = vdem_2000s_grouped_df)
print(f"Count of null records in the dataframe after cleaning: {vdem_2000s_grouped_df.isnull().sum()}")

#%%[markdown]
### 7.2 Check - Duplicate records

#%%
# checking duplicated values for both the dataframes 'vdem_2000s_df' & 'vdem_2000s_grouped_df'
print(f"Count of duplicated records in the 'vdem_2000s_df' dataframe: {vdem_2000s_df.duplicated().sum()}")
print(f"Count of duplicated records in the 'vdem_2000s_grouped_df' dataframe: {vdem_2000s_grouped_df.duplicated().sum()}")

#%%[markdown]
### 7.3 Check - Outliers

#%%
# Handling outliers 
def identify_outliers_tukey(df, column):
    """
    Identifying outliers in tukey method using IQR (Inter-Quartile range)

    Keyword arguments:
    df : Pandas dataframe
    column: Column name of the dataframe

    Returns:
    outliers: dictionary contains all outliers of numerical columns

    """
    # if df[column].dtype in ['float64', 'int64']:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    outlier_cutoff = 1.5 * iqr
    lower_bound = q1 - outlier_cutoff
    upper_bound = q3 + outlier_cutoff
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

outliers_dict_list = []

dataframes_list = [vdem_2000s_df, vdem_2000s_grouped_df]
for dataframe in dataframes_list:
    # identify outliers in all numeric columns using the Tukey method
    outliers_dict = {}
    for column in dataframe.select_dtypes(include=np.number).columns:
        outliers = identify_outliers_tukey(dataframe, column)
        if not outliers.empty:
            outliers_dict[column] = outliers

    # print the outlier results from all columns
    if not outliers_dict:
        print("No outliers found.")
    else:
        for column, outliers in outliers_dict.items():
            print(f"The outliers in column {column} are:\n{outliers}")
    outliers_dict_list.append(outliers_dict)
        
#%%
# Step 8: checking data types (and change if necessary)

print(vdem_2000s_grouped_df.dtypes) # As intended, all variables are floating values.

print(vdem_2000s_df.dtypes)
#%%

vdem_2000s_grouped_df['demo_index'] = vdem_2000s_grouped_df['demo_index'].astype(float) # changing 'demo_index' to float
vdem_2000s_grouped_df['geo_rgn_geo'] = vdem_2000s_grouped_df['geo_rgn_geo'].astype(int) # changing 'geo_rgn_geo' to int
vdem_2000s_grouped_df['geo_reg_polc_g'] = vdem_2000s_grouped_df['geo_reg_polc_g'].astype(int) # changing 'geo_reg_polc_g' to int
vdem_2000s_grouped_df['geo_reg_polc_g6c'] = vdem_2000s_grouped_df['geo_reg_polc_g6c'].astype(int) # changing 'geo_reg_polc_g6c' to int
vdem_2000s_grouped_df['infra_radio(n)_sets'] = vdem_2000s_grouped_df['infra_radio(n)_sets'].astype(int) # changing 'infra_radio(n)_sets' to int
vdem_2000s_grouped_df['demo_ttl_popln'] = vdem_2000s_grouped_df['demo_ttl_popln'].astype(int) # changing 'demo_ttl_popln' to int
vdem_2000s_grouped_df['demo_urbn_popln'] = vdem_2000s_grouped_df['demo_urbn_popln'].astype(int) # changing 'demo_urbn_popln' to int
vdem_2000s_grouped_df['demo_ttl_popln_wb'] = vdem_2000s_grouped_df['demo_ttl_popln_wb'].astype(int) # changing 'demo_ttl_popln_wb' to int
vdem_2000s_grouped_df['c_suc_coup_attp'] = vdem_2000s_grouped_df['c_suc_coup_attp'].astype(int) # changing 'c_suc_coup_attp' to int
vdem_2000s_grouped_df['c_coup_attp'] = vdem_2000s_grouped_df['c_coup_attp'].astype(int) # changing 'c_coup_attp' to int

print(vdem_2000s_grouped_df.dtypes)

"""
The goal of the data cleaning process is to prepare the dataset ready for further modeling and analysis. Starting with a checking for null values, it then replaces them with the appropriate values, looks for duplicate records, and identifies outliers. The Tukey method is used by the code to find outliers in all numerical columns.
The EDA procedure seeks to investigate the data and comprehend its properties. Data types are examined, made changes as necessary. The distribution of the variables, their correlations, and any additional relevant information are next checked.
Overall, the code offers a thorough EDA and data cleaning process that aids in getting the dataset ready for analysis and modeling.
"""
#%%
# A statistical summary of the numerical columns in the dataframe
summary = vdem_2000s_df.describe()
print(summary) #This will print summary statistics of vdem_2000s_df dataframe
summary_grouped = vdem_2000s_grouped_df.describe()

print(summary_grouped) #This will print summary statistics of vdem_2000s_grouped_df dataframe

#An overview of the dataframe's columns, including their names, data types, and non-null count.
vdem_2000s_df.info()
vdem_2000s_grouped_df.info()

#The dimensions of the dataframe
vdem_2000s_df.shape
vdem_2000s_grouped_df.shape

#%%
# Step 9: Test 2 (If anything goes wrong, just go back and check Step 5)



# %% measuing normality of demo_index in grouped vdem df

demo_index_grouped = tuple(vdem_2000s_grouped_df['demo_index'])

sns.displot(x=demo_index_grouped, bins=20)

#%% Boxplot
# The boxplot allows us to compare the distribution of democracy index across regions, as well as identify any potential outliers. It can also help us to identify if there are any significant differences in the median democracy index among different regions.

sns.boxplot(x="country_name", y="demo_index", data=vdem_2000s_grouped_df)

#%% Initial time-series line plot

# Set the 'year' column of the 'vdem_2000s_df' dataframe to an int data type. 
vdem_2000s_df['year'] = vdem_2000s_df['year'].astype(int)

# a list of sample countries to use it as input.
random_sample = ['North Korea', 'Denmark']

# Edge-case for randome sampling
def get_random_n_countries(col: str, n : int, sample: list) -> list:
    """
    This function is used to extract 3 random country names from dataframe
    
    Keyword arguments:
    col : column name (country_name)
    n: number of countries to extract
    sample : A list contains 2 countries as max and min limits
    """
    
    sample_countries = [] # Empty list to store n random country names
    while True:
        sample_countries = random.sample(vdem_2000s_grouped_df[country_var].unique().tolist(), n)

        if any(sample_country in sample for sample_country in sample_countries): # if country already exists in list, re-loop
            continue
        return sample_countries

country_var = "country_name"
sample_countries = get_random_n_countries(col = country_var, n = 3, sample = random_sample)

# Select a subset of the 'vdem_2000s_grouped_df' dataframe that contains only the rows corresponding to the randomly selected countries.
vdem_2000s_grouped_df_subset = vdem_2000s_grouped_df[vdem_2000s_grouped_df[country_var].isin(sample_countries)]

# It adds these country names to the 'random_sample' list. 
random_sample.extend(list(vdem_2000s_grouped_df_subset['country_name']))

# Select a subset of the 'vdem_2000s_df' dataframe that contains only the rows corresponding to the countries in the 'random_sample' list.
vdem_2000s_df_samples = vdem_2000s_df[vdem_2000s_df['country_name'].isin(random_sample)]

# Create a line plot of 5 countries which illustrates limits and comparing metrics. 
sns.lineplot(data=vdem_2000s_df_samples, x='year', y='democracy_index', hue='country_name')

#%% Scatterplot

# The cmap variable defines the color palette used for the plot.
cmap = sns.cubehelix_palette(rot=-2, as_cmap=True)
g = sns.relplot(
    data=vdem_2000s_df,
    x='democracy_index', y='e_civil_war',
    palette=cmap,
)
# The set() function is used to set the x-axis and y-axis scales to logarithmic scales.
g.set(xscale="log", yscale="log")
# The grid() function is used to add minor gridlines to the plot.
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
# The despine() function is used to remove the spines on the left and bottom sides of the plot.
g.despine(left=True, bottom=True)

plt.show()


#%% Scatterplot 2
# This scatterplot helps in visualizing the relationship between democracy index, total resources income per capita, region, and year.
# The x-axis represents the democracy index, the y-axis represents total resources income per capita, and the size of the dots represents the year. The color of the dots represents the region.
# The size of the dots adds an additional layer of information by showing how the data changes over time. The use of different colors helps to distinguish the different regions, making it easier to see if there are any regional patterns in the data.

cmap = sns.cubehelix_palette(rot=-2, as_cmap=True)
g = sns.relplot(
    data=vdem_2000s_df,
    x='democracy_index', y='e_total_resources_income_pc',
    hue='e_regionpol_6C', size='year',
    palette=cmap, sizes=(10,500),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

plt.show()

#%% Scatterplot 3
# This scatterplot shows the relationship between democracy index and GDP per capita, with points colored by the presence or absence of civil war (0 for no civil war, 1 for civil war) and sized by year.

# convert the e_civil_war column from a boolean to an integer using astype(int).
vdem_2000s_df['e_civil_war'] = vdem_2000s_df['e_civil_war'].astype(int)

cmap = sns.cubehelix_palette(rot=-2, as_cmap=True)
g = sns.relplot(
    data=vdem_2000s_df,
    x='democracy_index', y='e_gdppc',
    hue='e_civil_war', hue_order=[0, 1],
    size='year', sizes=(10,50),
    palette="muted", alpha=.5,
)
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

plt.show()

ax = plt.figure().add_subplot(projection='3d')

x = vdem_2000s_df['year']
y = vdem_2000s_df['democracy_index']
ax.plot(x, y, zs=0, zdir='z')

#%% Bubble plot animation (attempt #1)
# The  Bubble plot shows the relationship between democracy index and income inequality (measured by the Palma ratio) across different countries in the V-Dem dataset. The size of each bubble represents the country's GDP per capita, and the color represents the country name.

fig, ax = plt.subplots(figsize=(10,6))
sns.set(style="whitegrid")

def update(year):
    # Filter data by year
    data_year = vdem_2000s_df[vdem_2000s_df['year'] == year]

    # Updating plot
    ax.clear()
    sns.scatterplot(x='democracy_index', y='e_peedgini', data=vdem_2000s_df, size='e_gdppc', hue='country_name', sizes=(20, 2000), ax=ax)
    plt.title('Year: ' + str(year))

# Creating animation
animation = FuncAnimation(fig, update, frames=range(2000,2022, 1), repeat=True)

# saving animation as a GIF
writer = ani.PillowWriter(fps=1,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
animation.save('bubble.gif', writer=writer)

plt.show() # animation won't move here, have to open it in your working directory to see GIF

# %% Bubble plot animation (attempt #2)

fig = px.scatter(vdem_2000s_df, x='democracy_index', y='e_peedgini', size='e_gdppc', color='country_name', animation_frame='year', range_x=[0, 1], range_y=[0, 100], log_x=True, hover_name='country_name', size_max=60)

vdem_2000s_grouped_df_names['name'] = vdem_2000s_grouped_df['country_name']
vdem_2000s_grouped_df_names['region'] = vdem_2000s_grouped_df['geo_reg_polc_g6c']

#%% Bubble plot animation (attempt #3)

fig, ax = plt.subplots()

ax = vdem_2000s_grouped_map2.plot(column='region')
ax.set_axis_off()

ax.plot()

#%%[markdown]
# ### Basic EDA
# A heatmap of the correlation matrix for the vdem_2000s_grouped_df dataframe, highlighting cells where the correlation coefficient is less than -0.3.

corr = vdem_2000s_grouped_df.corr().abs() >= 0.7

# A mask is then created using the np.triu() function to exclude the upper triangle of the heatmap, as it is redundant due to symmetry.
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, annot = True, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

plt.show()

plt.show()

#%%
# Calculating the Variance Inflation Factor (VIF) for the columns 'demo_mrty(m)_rate' and 'demo_mrty(i)_rate' in the vdem_2000s_grouped_df dataframe. 

life_expect = vdem_2000s_grouped_df[['demo_mrty(m)_rate', 'demo_mrty(i)_rate']]

vif = pd.DataFrame()
# Creating a new dataframe life_expect that contains only the columns of interest.
vif["VIF Factor"] = [variance_inflation_factor(life_expect.values, i) for i in range(life_expect.shape[1])]
vif["Variable"] = life_expect.columns

vif

#%%[markdown]
## Interpreting the results of the basic EDA
# The basic EDA has generated a correlation matrix between the different variables in vdem_2000s_grouped_df. The heatmap shows the strength and direction of the correlation between different pairs of variables.
# The green shades with 1 written in the heatmap indicate a stronger positive correlation between the variables.
# The light blue shades with 0 in the heatmap indicate a stronger negative correlation between the variables.
# 

# Variables to test against average democracy index: GDP per capita, overall life expectancy, and average education.

#%%[markdown]
# ### Descriptive Statistics
print(vdem_2000s_grouped_df.columns)
# Select variables of interest
df = vdem_2000s_grouped_df[['demo_index', 'eco_gdp_pc', 'demo_life_expcy', 'edu_avg']]

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
#  

#%%[markdown]
# ### Hypothesis Testing(Correlation, Regression, etc.)

# Correlation
corr, p_value = pearsonr(df['demo_index'], df['eco_gdp_pc'])
if p_value < 0.05:
    print("There is a significant correlation between the average democracy index and GDP per capita.")
else:
    print("There is no significant correlation between the average democracy index and GDP per capita.")

# Linear regression
X = df[['eco_gdp_pc', 'demo_life_expcy', 'edu_avg']]
X = sm.add_constant(X)
y = df['demo_index']
model = sm.OLS(y, X).fit()
if model.f_pvalue < 0.05:
    print("There is a significant linear relationship between the average democracy index and the variables of interest.")
else:
    print("There is no significant linear relationship between the average democracy index and the variables of interest.")

#%%

#%%[markdown]
# Interpreting the results of the hypothesis testing
# The interpretation of these results depends on the direction and strength of the correlations/relationships. A significant correlation between the average democracy index and GDP per capita indicates that as GDP per capita increases, the average democracy index tends to increase as well. This could suggest that economic development is associated with increased political freedoms and democratic governance.
# A significant linear relationship between the average democracy index and the variables of interest suggests that there is a predictable pattern in the relationship between the two variables. For example, as the level of education increases, the average democracy index tends to increase as well. Similarly, as life expectancy increases, the average democracy index tends to increase. This could suggest that there are underlying factors, such as education and health, that contribute to the development of democratic governance.

#%% 
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA test
model = ols('demo_index ~ C(geo_rgn_geo)', data=vdem_2000s_grouped_df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

# Multiple Regression
model = ols('demo_index ~ eco_gdp_pc + demo_life_expcy + edu_avg + geo_rgn_geo', data=vdem_2000s_grouped_df).fit()
print(model.summary())
#%%[markdown]
# ### Correlation Analysis

#%%

#%%[markdown]
# Interpreting the results of the correlation analysis

#%%[markdown]
# ## Model Building

def country_name_encoded(df : pd.DataFrame, col : str) -> pd.DataFrame:
    # create a label encoder object
    le = LabelEncoder()

    # encode the "country_name" column as numerical values
    df['country_encoded'] = le.fit_transform(df[col])

    # drop the original "country_name" column
    df.drop(col, 
            axis = 1, 
            inplace = True)
    
    return df

# function for model building and mse
def random_forest_model(df : pd.DataFrame, target_col : str) -> float:
    """Function builds random forest model with vdem_2000s dataframe which splits
    data into train and test and returns RMSE value of the model.

    Keyword arguments:
    df -- A pandas dataframe of v-dem-2000s

    Return value:
    RMSE : A float value which represents root mean square error of R.F model
    """
    if 'country_encoded' not in df.columns:
        df = country_name_encoded(df, col = 'country_name')

    # Define the features and target
    y = df[target_col]
    X = df.drop([target_col], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 42)

    # Create the random forest regressor
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = rf.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = mean_squared_error(y_test, y_pred)

    # print("Random Forest MSE: {:.3f}".format(mse))
    return mse

target_variable = "democracy_index"
mse = random_forest_model(vdem_2000s_df, target_col = target_variable)
print(print("Random Forest MSE: {:.3f}".format(mse)))

#%%

#%%[markdown]
### Model interpretation
# The mean squared error (MSE) for a random forest model is showing as 0.0, it indicates that the model 
# is overfitting the data. In other words, the model is memorizing the training data instead of learning 
# the underlying patterns and relationships in the data.

#%%[markdown]
# ## Result

#%%[markdown]
# ## Discussion (Limitations, Future Work, etc.)
#%%[markdown]
# ## Conclusion

#%%[markdown]
# ## References

# %%
