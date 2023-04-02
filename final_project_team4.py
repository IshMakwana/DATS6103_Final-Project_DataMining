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
# Team Member: Daniel Felberg, Ei Tanaka, Ishani Makwana, Tharaka Maddineni

# ## Introduction

#%%
# This chunk is for set up modules and libraries
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
from scipy.stats import shapiro

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
VDem = getDFfromZip(url)
VDem.head()
VDem.shape

#%%[markdown]
# ## Data Cleaning and Prepareation
"""
Variables of interest (38 in totall):

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

(Gegraphy: 4 variables)
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
    "e_pt_coup_attempts"
]

# Select the desired columns from the DataFrame
vdem_df = VDem.loc[:, variables]
# Check the shape of the new DataFrame
print(vdem_df.shape)
vdem_df.head()

#%%
# Step 2: Create a new democracy index column at right to country_id column from 5 democracy variables
# Calculate the mean of the five democracy variables for each row
vdem_df['democracy_index'] = vdem_df[['v2x_polyarchy', 'v2x_libdem', 'v2x_partipdem', 'v2x_delibdem', 'v2x_egaldem']].mean(axis=1)
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
vdem_2000s_df = vdem_2000s_df.set_index(['country_name', 'country_id'])

# Group by 'country_id' and 'country_name', and aggregate the mean
vdem_2000s_grouped_df = vdem_2000s_df.groupby(['country_name', 'country_id']).agg("mean")

# Reset the index, so 'country_name' becomes a column again
vdem_2000s_grouped_df = vdem_2000s_grouped_df.reset_index()

# Remove the 'year' column
vdem_2000s_grouped_df = vdem_2000s_grouped_df.drop(columns=["year"])

# Display the combined DataFrame
print(vdem_2000s_grouped_df.shape)
vdem_2000s_grouped_df.head()
#%%[markdown]
# Dataframe variables we created so far(Step 1-4)
"""
VDem: original data set
vdem_df: subset containing only the columns of interest (38 variables) + democracy index
vdem_2000s_df: subset containing only 2000s in year column
vdem_2000s_grouped_df: combine the datasets by country (Combine multiple years into one and remove year column)
"""

#%%
# Step 5: Test 1 (If something is wrong, tell team members and go back to Step 1)

#%%
# Step 6:  change variable name
variable_names = ['country_name', 'country_id', 'year', 'elec_demo_idx', 'lib_demo_idx', 
                  'parti_demo_idx', 'deli_demo_idx', 'ega_demo_idx', 'edu_avg', 'edu_ineql', 
                  'geo_area', 'geo_reg_geo', 'geo_reg_pol_g', 'geo_reg_pol_g6c', 'eco_exports', 
                  'eco_imports', 'eco_gdp', 'eco_gdp_pc', 'eco_ifl_rate(a)', 'eco_popultn', 
                  'n_ttl_fuel_income_pc', 'n_ttl_oil_income_pc', 'n_ttl_res_income_pc', 
                  'infra_radio(num)_sets', 'e_miferrat', 'e_mipopula', 'e_miurbani', 'e_miurbpop', 
                  'e_pefeliex', 'e_peinfmor', 'e_pelifeex', 'e_pematmor', 'e_wb_pop', 'e_civil_war', 
                  'e_miinteco', 'e_miinterc', 'e_pt_coup', 'e_pt_coup_attempts']

vdem_2000s_grouped_df.columns = variable_names
print(vdem_2000s_grouped_df.shape)
print(vdem_2000s_grouped_df.head())

#%%[markdown]
## Step 7: Data Cleaning(drop null, drop duplicates, etc.)

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


# Checking null values on latest dataframe (vdem_2000s_grouped_df) 
# [Ask team members to check which dataframe needs to be cleaned]

print(f"Count of null records in the dataset before cleaning: {vdem_2000s_grouped_df.isnull().sum()}")

def fill_na(df):
    """
    This function replace the null values with mean or median value,
    within in the dataframe and returns the cleaned dataframe

    Keyword arguments:
    df : V-dem dataframe 

    Returns:
    cleaned_df : Clean dataframe
    """
    fill_null_0 = lambda col: col.fillna(0) if col.dtype in ['float64', 'int64'] else col.fillna(method='ffill')
    df = df.apply(fill_null_0)

    # Replacing null values with appropriate value (either mean or median)
    fill_null_imputation = lambda col: col.fillna(mean_median_imputation(column = col)) if col.dtype in ['float64', 'int64'] else col.fillna(method='ffill')
    cleaned_df = df.apply(fill_null_imputation)

    return cleaned_df

vdem_2000s_grouped_df = fill_na(df = vdem_2000s_grouped_df)

print(f"Count of null records in the dataset after cleaning: {vdem_2000s_grouped_df.isnull().sum()}")

#%%[markdown]
### 7.2 Check - Duplicate records

#%%
# checking duplicated values on latest dataframe (vdem_2000s_grouped_df)
print(f"Count of duplicated records in the dataset: {vdem_2000s_grouped_df.duplicated().sum()}")

#%%[markdown]
### 7.3 Check - Outliers

#%%
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

# identify outliers in all numeric columns using the Tukey method
outliers_dict = {}
for column in vdem_2000s_grouped_df.select_dtypes(include=np.number).columns:
    outliers = identify_outliers_tukey(vdem_2000s_grouped_df, column)
    if not outliers.empty:
        outliers_dict[column] = outliers

# print the outlier results from all columns
if not outliers_dict:
    print("No outliers found.")
else:
    for column, outliers in outliers_dict.items():
        print(f"The outliers in column {column} are:\n{outliers}")


#%%
# Step 8: check data type (and change if necessary)

#%%
# Step 9: Test 2

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