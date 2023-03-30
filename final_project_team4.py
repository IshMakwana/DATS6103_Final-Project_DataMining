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
VDem = getDFfromZip(url)
VDem.head()
VDem.shape

#%%[markdown]
# ## Data Cleaning and Prepareation

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
variable_names = ['country_name', 'country_id', 'year', 'elec_demo_idx', 'lib_demo_idx', 'parti_demo_idx', 'deli_demo_idx', 'ega_demo_idx', 'edu_avg', 'edu_ineql', 'geo_area', 'geo_reg_geo', 'geo_reg_pol_g', 'geo_reg_pol_g6c', 'eco_exports', 'eco_imports', 'eco_gdp', 'eco_gdp_pc', 'eco_ifl_rate(a)', 'eco_popultn', 'n_ttl_fuel_income_pc', 'n_ttl_oil_income_pc', 'n_ttl_res_income_pc', 'infra_radio(num)_sets', 'e_miferrat', 'e_mipopula', 'e_miurbani', 'e_miurbpop', 'e_pefeliex', 'e_peinfmor', 'e_pelifeex', 'e_pematmor', 'e_wb_pop', 'e_civil_war', 'e_miinteco', 'e_miinterc', 'e_pt_coup', 'e_pt_coup_attempts']
vdem_2000s_grouped_df.columns = variable_names
vdem_2000s_grouped_df.head
#%%
# Step 7: Data Cleaning(drop null, drop duplicates, etc.)

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