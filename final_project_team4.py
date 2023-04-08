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
# ## Data Cleaning and Preparation

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
vdem_2000s_df_indexed = vdem_2000s_df.set_index(['country_name', 'country_id'])

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
# 
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
# Step 6:  change variable name
variable_names = ['country_name', 'country_id', 'demo_index', 'elec_demo_idx', 'lib_demo_idx', 
                  'parti_demo_idx', 'deli_demo_idx', 'ega_demo_idx', 'edu_avg', 'edu_ineql', 
                  'geo_area', 'geo_rgn_geo', 'geo_reg_polc_g', 'geo_reg_polc_g6c', 'eco_exports', 
                  'eco_imports', 'eco_gdp', 'eco_gdp_pc', 'eco_a_ifl_rate', 'eco_popln', 
                  'n_ttl_fuel_income_pc', 'n_ttl_oil_income_pc', 'n_ttl_res_income_pc', 
                  'infra_radio(n)_sets', 'demo_frtly_rate', 'demo_ttl_popln', 'demo_urbzn_rate', 'demo_urbn_popln', 
                  'demo_lf_expcy(w)', 'demo_mrty(i)_rate', 'demo_life_expcy', 'demo_mrty(m)_rate', 'demo_ttl_popln_wb', 'c_civil_war', 
                  'c_intnl_arm_c', 'c_intl_arm_c', 'c_suc_coup_attp', 'c_coup_attp']
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
vdem_2000s_df = fill_na(df = vdem_2000s_df)


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
        
#%%[markdown]
## EDA

#%%
# Step 8: check data type (and change if necessary)

print(vdem_2000s_grouped_df.dtypes) # As intended, all variables are floating values.

print(vdem_2000s_df.dtypes)
#%%

vdem_2000s_grouped_df['demo_index'] = vdem_2000s_grouped_df['demo_index'].astype(float) # change 'democracy_index' to float
vdem_2000s_grouped_df['geo_rgn_geo'] = vdem_2000s_grouped_df['geo_rgn_geo'].astype(int) 
vdem_2000s_grouped_df['geo_reg_polc_g'] = vdem_2000s_grouped_df['geo_reg_polc_g'].astype(int)  
vdem_2000s_grouped_df['geo_reg_polc_g6c'] = vdem_2000s_grouped_df['geo_reg_polc_g6c'].astype(int) 
vdem_2000s_grouped_df['infra_radio(n)_sets'] = vdem_2000s_grouped_df['infra_radio(n)_sets'].astype(int) 
vdem_2000s_grouped_df['demo_ttl_popln'] = vdem_2000s_grouped_df['demo_ttl_popln'].astype(int) 
vdem_2000s_grouped_df['demo_urbn_popln'] = vdem_2000s_grouped_df['demo_urbn_popln'].astype(int) 
vdem_2000s_grouped_df['demo_ttl_popln_wb'] = vdem_2000s_grouped_df['demo_ttl_popln_wb'].astype(int) 
vdem_2000s_grouped_df['c_suc_coup_attp'] = vdem_2000s_grouped_df['c_suc_coup_attp'].astype(int) 
vdem_2000s_grouped_df['c_coup_attp'] = vdem_2000s_grouped_df['c_coup_attp'].astype(int) 

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

#%%[markdown]
# ## EDA


# %% measuing normality of demo_index in grouped vdem df

demo_index_grouped = tuple(vdem_2000s_grouped_df['demo_index'])

sns.displot(x=demo_index_grouped, bins=20)

#%% Boxplot

sns.boxplot(x="country_name", y="demo_index", data=vdem_2000s_grouped_df)

#%% Initial time-series line plot

vdem_2000s_df['year'] = vdem_2000s_df['year'].astype(int)

country_var = "country_name"

sample_countries = random.sample(vdem_2000s_grouped_df[country_var].unique().tolist(), 3)

vdem_2000s_grouped_df_subset = vdem_2000s_grouped_df[vdem_2000s_grouped_df[country_var].isin(sample_countries)]

random_sample = list(vdem_2000s_grouped_df_subset['country_name'])
random_sample.append('North Korea')
random_sample.append('Denmark')

vdem_2000s_df_samples = vdem_2000s_df[vdem_2000s_df['country_name'].isin(random_sample)]

sns.lineplot(data=vdem_2000s_df_samples, x='year', y='democracy_index', hue='country_name')

#%% Scatterplot

cmap = sns.cubehelix_palette(rot=-2, as_cmap=True)
g = sns.relplot(
    data=vdem_2000s_df,
    x='democracy_index', y='e_civil_war',
    size='',
    palette=cmap,
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

#%% Scatterplot 2

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

#%% Scatterplot 3

vdem_2000s_df['e_civil_war'] = vdem_2000s_df['e_civil_war'].astype(int)

cmap = sns.cubehelix_palette(rot=-2, as_cmap=True)
g = sns.relplot(
    data=vdem_2000s_grouped_df,
    x='demo_index', y='eco_gdp_pc',
    size='c_civil_war',
    palette=cmap, sizes=(10,500),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

#%% Another time-series line plot attempt

ax = plt.figure().add_subplot(projection='3d')

x = vdem_2000s_df['year']
y = vdem_2000s_df['democracy_index']
ax.plot(x, y, zs=0, zdir='z')

#%% Bubble plot animation (attempt #1)

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

fig.show()

#%% Bubble plot animation (attempt #3)

fig, ax = plt.subplots()

scat = ax.scatter(1,0)
x = vdem_2000s_df['democracy_index']

def animate(i):
    scat.set_offsets((x[i], 0))
    return scat,

ani = animation.FuncAnimation(fig, animate, repeat=True,
                              frames=vdem_2000s_df['year'] - 1, interval=50)

writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('bubble.gif', writer=writer)

plt.show()

#%%[markdown]
# ### Basic EDA

corr = vdem_2000s_grouped_df.corr() < -0.3

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%

life_expect = vdem_2000s_grouped_df[['demo_lf_expcy(w)', 'demo_life_expcy']]

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(life_expect.values, i) for i in range(life_expect.shape[1])]
vif["Variable"] = life_expect.columns

#%%[markdown]
# Interpreting the results of the basic EDA

# Variables to test against average democracy index: GDP per capita, overall life expectancy, and average education.

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