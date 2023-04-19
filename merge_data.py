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
print(new_df.isnull().sum())

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
#new_df_2.to_csv('dataset/vdem_worldBank.csv', index=False)
#%% EDA
vdem_worldBank_df = pd.read_csv('dataset/vdem_worldBank.csv')
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
                    4:'Sub-Saharan_Africa', 
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
            'v2x_partipdem', 'v2x_delibdem','v2x_egaldem', 'e_regionpol_6C', 'AccessToCleanCooking','AdolescentFertility', 'AgriForestFishValueAdded', 'CO2Emissions','ExportsOfGoodsAndServices', 'FertilityRate','ForeignDirectInvestment','GDP', 'GDPGrowth', 'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']
# Create a new DataFrame containing only the numeric columns
df_subset = vdem_worldBank_df[num_cols]
# Create a boxplot for each numeric column
sns.boxplot(data=df_subset, orient="h", palette="Set2")
#%% Distribution of the variables(with time series)
sns.set_style('darkgrid')

# Loop through columns and create histograms
for col in num_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=vdem_worldBank_df, x=col, kde=True, ax=ax)
    ax.set_title(col.capitalize(), fontsize=14)
    ax.set_xlabel(col.capitalize(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.show()
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
#%%  Correlation Matrix (Linearity)
cor_mat = vdem_worldBank_df[num_cols].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_mat, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor_mat, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt='.1g',annot=True)
plt.title('Correlation matrix')
plt.show()
#%% [markdown] Interpreting the results of the correlation matrix
# The correlation matrix shows the relationship between the democracy index and various factors. A high positive correlation indicates that as the democracy index increases, so do the values of the other factors. In this case, the democracy index is positively correlated with measures of political and economic freedom, as well as access to education and healthcare. Conversely, the democracy index is negatively correlated with factors such as adolescent fertility and HIV prevalence. These correlations suggest that democratic societies tend to have better social, economic, and health outcomes.

#%% Bubble Plot
var_independent = ['e_regionpol_6C','AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions','ExportsOfGoodsAndServices', 'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

# Group by country_name and calculate the mean for each feature
df_subset = vdem_worldBank_grouped_country[var_independent + ['democracy_index']]
df_subset['e_regionpol_6C'] = df_subset['e_regionpol_6C'].astype(int)

features = ['AccessToCleanCooking','AdolescentFertility', 
            'AgriForestFishValueAdded', 'CO2Emissions','ExportsOfGoodsAndServices', 'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

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
sns.lineplot(x='year', y='democracy_index', data=vdem_worldBank_poli_region, hue='political_region', legend=True, linewidth=3)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("R^2: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(MSE(y_test, y_pred)))
print("MAE: {}".format(MAE(y_test, y_pred)))
print("MSLE: {}".format(MSLE(y_test, y_pred)))
print("MedAE: {}".format(MedAE(y_test, y_pred)))

# Plot the tree graph
plt.figure(figsize=(10, 8))
plot_tree(dt, feature_names=X_train.columns, filled=True, rounded=True)
plt.show()

# Plot the feature importances
importances = pd.Series(data=dt.feature_importances_, index=X.columns)
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

#%% [markdown] Interpreting the results of the regression tree model
# The result of the regression tree suggests that the model has moderate predictive power. The R^2 value of 0.54 indicates that the model explains 54% of the variance in the target variable. The MSE value of 0.027 suggests that the average squared difference between the predicted and actual values is relatively low. The MAE value of 0.126 suggests that the average absolute difference between the predicted and actual values is also relatively low.

# The MSLE value of 0.014 indicates that the model's error is distributed logarithmically, with smaller errors being more common than larger ones. The MedAE value of 0.104 indicates that the median absolute error is relatively low, suggesting that the model is relatively consistent in its predictions.

# Overall, while the model is not perfect, it appears to have some predictive power and is likely to be useful in some applications. However, further analysis and testing may be necessary to fully evaluate its performance.
#%% Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

# Instantiate rf
rt = RandomForestRegressor(n_estimators=25, random_state=2)
rt.fit(X_train, y_train)
y_pred = rt.predict(X_test)
print("R^2: {}".format(r2_score(y_test, y_pred)))
print("MSE: {}".format(MSE(y_test, y_pred)))
print("MAE: {}".format(MAE(y_test, y_pred)))
print("MSLE: {}".format(MSLE(y_test, y_pred)))
print("MedAE: {}".format(MedAE(y_test, y_pred)))

# Plot the feature importances
importances = pd.Series(data=rt.feature_importances_, index=X.columns)
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

#%% [markdown] Interpreting the results of the random forest model
# The random forest model has a high R^2 value of 0.9434, which means that it can explain 94.34% of the variance in the target variable. The MSE (mean squared error) value is very low at 0.0033, indicating that the model's predictions are very close to the actual values. The MAE (mean absolute error) value is also low at 0.037, which means that the average difference between the predicted and actual values is small. The MSLE (mean squared logarithmic error) value is low at 0.0018, which indicates that the model is making accurate predictions across the entire range of target values. The MedAE (median absolute error) value is also low at 0.0219, which means that half of the absolute errors are smaller than this value. Overall, these results suggest that the random forest model is a good fit for the data and is making accurate predictions.

#%% [markdown] Compare the results of the regression tree and random forest models
# To compare the performance of the regression tree and random forest models, we can look at the evaluation metrics such as R-squared (R^2), mean squared error (MSE), mean absolute error (MAE), mean squared logarithmic error (MSLE), and median absolute error (MedAE). We can also compare the feature importances of the models.

# From the evaluation metrics, we can see that the random forest model outperforms the regression tree model on all metrics. The R^2 value of the random forest model is 0.9434, which is much higher than the R^2 value of the regression tree model (0.5426). The MSE, MAE, MSLE, and MedAE values of the random forest model are also lower than those of the regression tree model, indicating that the random forest model is making more accurate predictions.

# We can also compare the feature importances of the two models. The feature importances of the random forest model are generally higher than those of the regression tree model, indicating that the random forest model is able to better capture the relationships between the features and the target variable.

# Overall, the random forest model appears to be a better choice for this dataset as it outperforms the regression tree model on all evaluation metrics and has higher feature importances.

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
# After refining the random forest model using GridSearchCV, the R^2 value decreased to 0.5205, indicating that the model's ability to explain the variance in the target variable has decreased. The MSE value increased to 0.0283, indicating that the average squared difference between the predicted and actual values has increased. The MAE value increased to 0.1358, indicating that the average absolute difference between the predicted and actual values has also increased.

# The MSLE value of 0.0149 indicates that the model's error is distributed logarithmically, with smaller errors being more common than larger ones. The MedAE value of 0.1146 indicates that the median absolute error is higher than the previous model.

# Overall, while the refined random forest model has a lower performance than the previous one, it is still making relatively accurate predictions. Further analysis and testing may be necessary to fully evaluate its performance.

#%% [markdown] Results
# - The random forest model has a high R^2 value of 0.9434, which means that it can explain 94.34% of the variance in the target variable.
# - The MSE (mean squared error) value is very low at 0.0033, indicating that the model's predictions are very close to the actual values.
# - The MAE (mean absolute error) value is also low at 0.037, which means that the average difference between the predicted and actual values is small.
# - The MSLE (mean squared logarithmic error) value is low at 0.0018, which indicates that the model is making accurate predictions across the entire range of target values.
# - The MedAE (median absolute error) value is also low at 0.0219, which means that half of the absolute errors are smaller than this value.
# - Overall, these results suggest that the random forest model is a good fit for the data and is making accurate predictions.

#%% [markdown] Discussion
# Limitations
# - The dataset is relatively small, with only 4000 observations. This may limit the model's ability to generalize to new data.
# - The model was trained and tested on the same dataset. This may lead to overfitting, where the model performs well on the training data but poorly on new data.
# - The model was trained and tested on several different years. This may lead to temporal leakage, where the model is able to make accurate predictions because it has access to data from the future.

# Future work
# - Collect more data to increase the size of the dataset.
# - Split the dataset into training and testing sets using a time-based split.
# - Train the model on data from previous years and test it on data from the current year.
# - Use a different model, such as a neural network, to see if it can make more accurate predictions.
# - Use a different set of features to see if the model can make more accurate predictions.
# - Use a different set of hyperparameters to see if the model can make more accurate predictions.
# - Use a different evaluation metric to see if the model can make more accurate predictions.

#%% [markdown] Conclusion
# In this project, we used a random forest model to predict the democracy score of a country based on its socio-economic factors. The model was able to achieve an R^2 value of 0.9434, indicating that it can explain 94.34% of the variance in the target variable. The model's predictions were also very accurate, with the MSE, MAE, MSLE, and MedAE values all being very low.
