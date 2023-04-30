"""
Authors: Daniel Felberg, Ei Tanaka, Ishani Makwana, Tharaka Maddineni 
Data: 
Purpose: DATS6103 Final Project
"""

#%%[markdown]
## Data Mining Project-Team4
## Project Title: 21st Century Democracy Index Matrices
# Team Member: Daniel Felberg, Ei Tanaka, Ishani Makwana, Tharaka Maddineni

# ## Introduction
# Our project is comparing metrics used to determine democracy level  in different countries during the 21st century. The data set was created by V-Dem with 27,555 rows and 808 variables, which measures how democratic countries are using indices, based on numerous factors, including censorship, media bias, political repression, relationship between branches of government, equal rights, and access to democratic processes. The data set also includes “background factors” such as economic, education and other socioeconomic variables.
# The Varieties of Democracy (V-Dem) dataset is a comprehensive tool that allows for a more nuanced understanding of democracy by measuring and analyzing multiple dimensions of democratic practices and institutions. The dataset is based on the idea that democracy is not just about holding free and fair elections, but also about having institutions and practices that promote civil liberties, political equality, citizen participation, and deliberation among different groups.
# To measure these different aspects of democracy, the V-Dem project identifies five high-level principles: electoral democracy, liberal democracy, participatory democracy, deliberative democracy, and egalitarian democracy. These principles are further broken down into more specific indicators and sub-indicators, such as the fairness of electoral processes, the protection of civil liberties, the inclusion of marginalized groups, and the ability of citizens to participate in decision-making processes.
# By using this multidimensional and disaggregated dataset, researchers and policymakers can gain a more nuanced understanding of the strengths and weaknesses of democratic practices and institutions in different countries and regions, and identify areas for improvement.

#%% Chunk is for set up modules and libraries
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
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error


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

#%% [markdown]
#%% 
### Data collection
# Reading merged (V-dem + world-Bank) dataset from GitHub repository
url2 = "https://raw.githubusercontent.com/IshMakwana/DATS6103_Final-Project_DataMining/main/dataset/vdem_worldBank.csv"
vdem_worldBank_df = pd.read_csv(url2)

#%% Basic Information (vdem_worldBank_df)
print(vdem_worldBank_df.head())
print(vdem_worldBank_df.info())
print(vdem_worldBank_df.describe())
print(vdem_worldBank_df.columns)
vdem_worldBank_df['country_name'].unique() 

#%% Create a new dataframe aggregating the mean of the democracy index for each country
vdem_worldBank_grouped_country = vdem_worldBank_df.groupby('country_name').mean()
vdem_worldBank_grouped_country.reset_index(inplace=True)

#%% Basic Information (vdem_worldBank_grouped_country)
print(vdem_worldBank_grouped_country.head())
print(vdem_worldBank_grouped_country.info())
print(vdem_worldBank_grouped_country.isnull().sum())
print(vdem_worldBank_grouped_country.describe())

#%% Create a new dataframe aggregating the mean of the democracy index for each political region with time series
vdem_worldBank_poli_region = vdem_worldBank_df.groupby(['e_regionpol_6C', 'year']).mean()
vdem_worldBank_poli_region.reset_index(inplace=True)

#%% Basic Information (vdem_worldBank_poli_region)
print(vdem_worldBank_poli_region.head())
print(vdem_worldBank_poli_region.info())
print(vdem_worldBank_poli_region.describe())

#%% Create a new dataframe aggregating the mean of the democracy index for each political region
vdem_worldBank_poli_region_grouped = vdem_worldBank_df.groupby('e_regionpol_6C').mean()
vdem_worldBank_poli_region_grouped.reset_index(inplace=True)

#%% Basic Information (vdem_worldBank_poli_region_grouped)
print(vdem_worldBank_poli_region_grouped.head())
print(vdem_worldBank_poli_region_grouped.info())
print(vdem_worldBank_poli_region_grouped.describe())

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

#%% [markdown] 
### Distribution of the variables(with time series)

# Create histograms for each numeric column
sns.set_style('darkgrid')
for col in num_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=vdem_worldBank_df, x=col, kde=True, ax=ax)
    ax.set_title(col.capitalize(), fontsize=14)
    ax.set_xlabel(col.capitalize(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.show()

#%% [markdown] Interpreting the distribution plots
# The distributions of the numerical variables in the dataset have varying degrees of skewness. Some of the variables such as foreign direct investment, GDP growth rate, net migration, and population growth rate exhibit normal distribution, while others like adolescent fertility, CO2 emissions, and mortality rate are right-skewed. Democracy index, V-Dem scores, access to clean cooking, and mobile subscriptions are binomially distributed. Understanding the distribution of the variables can help in selecting appropriate statistical methods and interpreting the results accurately.

# Normal Distribution
# - foreign direct investment
# - GDP Growth Rate
# - Net Migration
# - Population Growth Rate
# - Primary School Enrollment
#
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
#
# Binomial Distribution
# - Democracy Index
# - V-Dem Polyarchy
# - V-Dem liberal democracy
# - V-Dem participatory democracy
# - V-Dem deliberative democracy
# - V-Dem egalitarian democracy
# - Access to Clean Cooking (% of population)
# - Mobile Subscriptions (per 100 people)

#%% [markdown]
# ### Box plot Analysis

#%% Boxplot
ax = sns.boxplot(data=vdem_worldBank_df, y="democracy_index", x="e_regionpol_6C", dodge=False)

ax.set_xticklabels(['E_Europe_Ctrl_Asia', 'Lt_America_Caribbean', 'MidEast_N_Africa', 'Sub_Saharan_Africa', 'W_Europe_NA_Oceania', 'Asia_Pacific'])
plt.xticks(rotation=90)
plt.show()

#%% [markdown]
# ### Bubble Plot Analysis
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

#%% [markdown] Interpreting the results of the time series analysis

#%% [markdown]
# ### Small Multiple Time Series Analysis

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

#%% [markdown] 
# ### Time Series Analysis
#%% Time series by political Region
# Democracy Index
sns.set_context("paper")
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'democracy_index', 
            hue =  'political_region', kind = 'line')
plt.show()

#%% Time series 2
# Life Expectancy
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'LifeExpectancy', 
            hue =  'political_region', kind = 'line')
plt.show()

#%% Time series 3
# Under 5 Mortality
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'Under5Mortality', 
            hue =  'political_region', kind = 'line')
plt.show()

#%% Time series 4
# GNI per capita
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'GNIPerCapita', 
            hue =  'political_region', kind = 'line')
plt.show()

#%% Time series 5
# Primary School Enrollment
sns.relplot(data = vdem_worldBank_df, 
            x = 'year', y = 'PrimarySchoolEnrollment', 
            hue =  'political_region', kind = 'line')
plt.show()

#%%[markdown]
# ### 3-D Scatter Plot Analysis
#%% 3-d scatter plot 1
# Democracy Index vs Life Expectancy vs Year
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

#%% 3-d scatter plot 2
# Democracy Index vs GNI Per Capita vs Year
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

#%% [markdown]
# ### Hyptothesis Testing
# ANOVA
# t-test

#%%
# Annova Test 3rd attempt
from scipy.stats import f_oneway
from scipy import stats

nan_count = vdem_worldBank_grouped_country.isna().sum()
print(nan_count)

ANOVA_variables = ['e_regionpol_6C','AccessToCleanCooking','AdolescentFertility', 
                   'AgriForestFishValueAdded', 'CO2Emissions', 'ExportsOfGoodsAndServices', 
                   'FertilityRate', 'ForeignDirectInvestment','GDP', 'GDPGrowth', 
                   'GNIPerCapita', 'MeaslesImmunization','ImportsOfGoodsAndServices', 
                   'LifeExpectancy', 'MobileSubscriptions','Under5Mortality', 'NetMigration', 
                   'PopulationGrowth', 'HIVPrevalence','PrimarySchoolEnrollment']

for var in ANOVA_variables:
    f_stat, p_val = stats.f_oneway(*[vdem_worldBank_grouped_country.loc[vdem_worldBank_grouped_country[var]==i, 'democracy_index'] 
                                      for i in vdem_worldBank_grouped_country[var].unique()])
    print(f"ANOVA test result for {var}:")
    print(f"F value: {f_stat:.2f}, p-value: {p_val:.4f}\n")


#%% [markdown]
### Feature Selection
# We used the filter method to perform feature selection. 
# Target variables and independent variables are both continuous numerical variables, 
# so we used pearson's Since both target variables and independent variables are continuous 
# numerical variables, we used variables with a correlation coefficient greater than or 
# equal to 0.3 or less than -0.3 as features based on the results of pearson's correlation.

### Correlation Matrix
#%%  Correlation Matrix (Linearity)
new_vdem_worldBank_df = vdem_worldBank_df[["democracy_index"] + features]

cor_mat = new_vdem_worldBank_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_mat, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(cor_mat, mask = mask, cmap = cmap, vmax = .3, center=0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5}, 
            fmt = '.1g', annot = True)

plt.title('Correlation matrix')
plt.show()

#%% [markdown] Interpreting the results of the correlation matrix

# The correlation matrix shows the relationship between the democracy index and various factors. 
# A high positive correlation indicates that as the democracy index increases, so do the values of the other factors. 
# In this case, the democracy index is positively correlated with measures of political and economic freedom, 
#   as well as access to education and healthcare. Conversely, the democracy index is negatively correlated with factors 
#   such as adolescent fertility and HIV prevalence. 
# These correlations suggest that democratic societies tend to have better social, economic, and health outcomes.


#%% Create a list of variables with correlations greater than or equal to 0.3 or less than or equal to -0.3
# Subset correlation matrix for variables with correlations greater than or equal to 0.3 or less than or equal to -0.3
high_corr = cor_mat[(cor_mat['democracy_index'] >= 0.3) | (cor_mat['democracy_index'] <= -0.3)]

# Get unique variable names
high_cor_features = high_corr.index.tolist()
high_cor_features = high_cor_features[1:]
print((high_cor_features))

#%%[markdown]
### Dataset Variance Inflation factor

def calc_vif(df, features):
    X = df[features]
    vif = pd.DataFrame({'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 'Features': features})
    print(vif)
    print()
calc_vif(vdem_worldBank_df, high_cor_features)

for feature in ['LifeExpectancy', 'FertilityRate']:
    high_cor_features.remove(feature)
    print(f"VIF after removing {feature} feature: ")
    calc_vif(vdem_worldBank_df, high_cor_features)

#%% [markdown] 
# ##Model Building
# Regression Tree Model
# Random Forest Model
# Gradient Boosting Model

#%%[markdown] 
## Model Building
### Model - 1: Decision (Regression) Tree

from sklearn.model_selection import train_test_split

# Using highly correlated features from the correlation matrix and significance testing

y = vdem_worldBank_df['democracy_index']
X = vdem_worldBank_df[high_cor_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.tree import plot_tree

# Instantiate the decision tree regressor model
dt_model = DecisionTreeRegressor(max_depth = 4, 
                                min_samples_leaf = 0.1, 
                                random_state = 3)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Decision/Regression Tree model (dt_model) evaluation metrics
print("Decision/Regression Tree model evaluation metrics: ")
r2_dt = r2_score(y_test, y_pred)
mse_dt = mean_squared_error(y_test, y_pred)
mae_dt = mean_absolute_error(y_test, y_pred)
msle_dt = mean_squared_log_error(y_test, y_pred)
medae_dt = median_absolute_error(y_test, y_pred)

# Evaluate the refined Random forest model
print(f"R^2: {r2_dt} ({round(r2_dt * 100, 1)}%)")
print(f"MSE: {mse_dt} ({round(mse_dt * 100, 1)}%)")
print(f"MAE: {mae_dt} ({round(mae_dt * 100, 1)}%)")
print(f"MSLE: {msle_dt} ({round(msle_dt * 100, 1)}%)")
print(f"MedAE: {medae_dt} ({round(medae_dt * 100, 1)}%)")

# Decision (Regression) Tree map
plt.figure(figsize = (12, 10))
plot_tree(dt_model, feature_names = X_train.columns, 
        filled = True, rounded = True, fontsize = 11,
        impurity = True, 
        proportion = True, 
        precision = 2)
plt.title('Decision Tree for variables of interest')
plt.show()

importances = pd.Series(data = dt_model.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()

# Plotting the bar histogram with labels and legends
importances_sorted.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.legend(['Feature Importance'], loc='lower right')

plt.show()

#%%[markdown]
#### Interpreting the results of the regression tree model:
#
# The result of the regression tree suggests that the model has moderate predictive power. 
# 
# The R^2 value of 0.49 indicates that the model explains 49% of the variance in the target variable. 
# 
# The MSE value of 0.0304 suggests that the average squared difference between the predicted and actual values is relatively low. 
#
# The MAE value of 0.138 suggests that the average absolute difference between the predicted and actual values is also relatively low.
#
# The MSLE value of 0.016 indicates that the model's error is distributed logarithmically, 
#     with smaller errors being more common than larger ones. 
#
# The MedAE value of 0.12 indicates that the median absolute error is relatively low, suggesting that the model is relatively consistent in its predictions.
#
# Overall, while the model is not perfect, it appears to have some predictive power and is likely to be useful in some applications. 
#
# However, further analysis and testing may be necessary to fully evaluate its performance.


#%%[markdown]
### Prediction accuracy with new data for DT model
# Hold-out concept (out-of-bag samples)

# dt_out_of_bag = RandomForestRegressor(oob_score = True)
# rf_out_of_bag.fit(X_train, y_train)
# rf_obb_score = rf_out_of_bag.oob_score_
# print(f"Prediction accuracy with new data: {round(rf_obb_score, 2) * 100}% ") 


#%%[markdown]

### Decision Tree Model performance

# Cross validation using sklearn
from sklearn.model_selection import cross_val_score

reg_tree_scores = cross_val_score(dt_model, 
                        X, y, cv = 5, 
                        scoring='r2')
scores = cross_val_score(dt_model, 
                        X, y, cv = 5, 
                        scoring='r2')

print("5-fold X-Validation for regression model:")
print("Cross-validation R^2 scores:")
for r_2 in reg_tree_scores:
    print(round(r_2, 4), end = ", ")
print()
print("Mean R^2:", round(reg_tree_scores.mean(), 4))


#%%[markdown]

#### Interpreting the results of the regression tree model-cross_validation metrics:
#
# The cross-validation R2 values are: 0.40813912 0.29964666 0.35225868 0.32715057 0.03348702.
#
# The R-squared value of the original regression model is 0.491, which indicates that the model 
#           explains about 49.1% of the variance in the dependent variable. 
#
# The R-squared value from the 5-fold cross-validation for the regression tree model is 0.284, 
#           which means that the model explains 28.4% of the variance on average across the 5 folds.
# 
# This suggests that the original model may be overfitting the data, and the performance of the 
#            model may not generalize well to new data. 
# 
# The cross-validation results provide a more accurate estimate of the model's performance on 
#            unseen data by evaluating the model on multiple subsamples of the data. 
# 
# Therefore, it may be necessary to consider modifying the model or performing further feature engineering 
#            to improve the model's performance.
#
# An improved model fit is generally indicated by a higher R2 value, 
# but it's also crucial to make sure the model isn't overfitting the data. 
# Cross-validation can give a more precise assessment of the model's performance and aid in evaluating the model's performance on unknown data.


#%%[markdown]
## Ensembling methods
### Model - 2
# Random Forest Model

#%% Ensembling methods / Random Forest Model

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectFromModel

# Instantiate rf
rf_model = RandomForestRegressor(n_estimators = 25, 
                                random_state = 2)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# RF model evaluation metrics
print("Random forest model evaluation metrics: ")
r2_rf = r2_score(y_test, rf_y_pred)
mse_rf = mean_squared_error(y_test, rf_y_pred)
mae_rf = mean_absolute_error(y_test, rf_y_pred)
msle_rf = mean_squared_log_error(y_test, rf_y_pred)
medae_rf = median_absolute_error(y_test, rf_y_pred)

print(f"R^2: {r2_rf} ({round(r2_rf * 100, 1)}%)")
print(f"MSE: {mse_rf} ({round(mse_rf * 100, 1)}%)")
print(f"MAE: {mae_rf} ({round(mae_rf * 100, 1)}%)")
print(f"MSLE: {msle_rf} ({round(msle_rf * 100, 1)}%)")
print(f"MedAE: {medae_rf} ({round(medae_rf * 100, 1)}%)")

#%%
# Plot the feature importances
importances = pd.Series(data = rf_model.feature_importances_, index = X.columns)
importances_sorted = importances.sort_values()

# Plotting the bar histogram with labels and legends
importances_sorted.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.legend(['Feature Importance'], loc='lower right')

plt.show()

#%%[markdown]

### Interpreting the results of the random forest model:

# The random forest model has a high R^2 value of 0.9304, 
#       which means that it can explain 93.4% of the variance in the target variable. 
# 
# The MSE (mean squared error) value is very low at 0.0041, indicating that the model's predictions 
#      are very close to the actual values. 
# 
# The MAE (mean absolute error) value is also low at 0.0452, which means that the average difference between the predicted and actual values is small. 
# 
# The MSLE (mean squared logarithmic error) value is low at 0.0023, which indicates that the model is making accurate predictions across the entire range of target values. 
# 
# The MedAE (median absolute error) value is also low at 0.0303, which means that half of the absolute errors are smaller than this value. 


#%%[markdown]
### Prediction accuracy with new data
# Hold-out concept (out-of-bag samples)

rf_out_of_bag = RandomForestRegressor(oob_score = True)
rf_out_of_bag.fit(X_train, y_train)
rf_obb_score = rf_out_of_bag.oob_score_
print(f"Prediction accuracy with new data: {round(rf_obb_score, 2) * 100}% ") 

#%%[markdown]
### Interpretation of model

# The code performs random forest regression on training data using the 'RandomForestRegressor()' function 
# with oob_score = True to enable 'out-of-bag' (OOB) scoring. 
#
# This technique uses some training data that is not used for training each individual tree to estimate 
# the accuracy of the model without requiring a separate test set or cross-validation.
#
# The model is trained using the fit() method on the training data, and the 'oob_score_' attribute 
# of the RandomForestRegressor object is used to display the OOB score of the model. 
#
# The output value of 'oob_score_' is '0.9108', indicating a high accuracy of the model on the OOB data, 
# with a score of '0.91' or '91%'. This suggests that the model is likely to perform well on new, 
# unseen data and may be suitable for generalization to such data.


#%%[markdown]

### Random forest Model performance [CV]    
# Cross validation using sklearn
from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(rf_model, 
                        X, y, cv = 5, 
                        scoring='r2')

print("5-fold X-Validation for random forest model:")
print("Cross-validation R^2 scores:")
for r_2 in rf_scores:
    print(round(r_2, 4), end = ", ")
print()
print("Mean R^2:", round(rf_scores.mean(), 4))

#%%[markdown]

#### The 5-fold cross-validation result for a random forest model is displayed in the output. 

# The cross-validation's R-squared values for each fold are: 0.47413435  0.3568414   0.38703758  0.53499121 -0.11419026
#   indicating that the model's performance differs considerably between folds. 
#
# Lower than the R-squared value of the initial random forest model, the mean R-squared value 
#   across all folds is 0.3277. 
#
# This implies that the original model may have overfitted the data and that the performance 
#   of the model may not be as excellent when applied to fresh, unforeseen data. 
#
# To boost the model's performance, it might be required to further tweak its hyperparameters 
#
#               or 
#
#   look into alternative machine learning techniques.

#%%[markdown]

###  Refining the random forest model
#%%Hyperparameter tuning

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
rf_best_y_pred = rf_best.predict(X_test)


print("Refined RF model evaluation metrics: ")
r2_rf_best = r2_score(y_test, rf_best_y_pred)
mse_rf_best = mean_squared_error(y_test, rf_best_y_pred)
mae_rf_best = mean_absolute_error(y_test, rf_best_y_pred)
msle_rf_best = mean_squared_log_error(y_test, rf_best_y_pred)
medae_rf_best = median_absolute_error(y_test, rf_best_y_pred)

# Evaluate the refined Random forest model
print(f"R^2: {r2_rf_best} ({round(r2_rf_best * 100, 1)}%)")
print(f"MSE: {mse_rf_best} ({round(mse_rf_best * 100, 1)}%)")
print(f"MAE: {mae_rf_best} ({round(mae_rf_best * 100, 1)}%)")
print(f"MSLE: {msle_rf_best} ({round(msle_rf_best * 100, 1)}%)")
print(f"MedAE: {medae_rf_best} ({round(medae_rf_best * 100, 1)}%)")
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

#%% [markdown] 
#### Interpreting the results of the refined random forest model
# After refining the random forest model using GridSearchCV, 

# the R^2 value decreased to 0.5119, indicating that the model's ability to explain the variance in the target variable has decreased. 
# The MSE value increased to 0.0291, indicating that the average squared difference between the predicted and actual values has increased.
# The MAE value increased to 0.1436, indicating that the average absolute difference between the predicted and actual values has also increased.
# The MSLE value of 0.0155 indicates that the model's error is distributed logarithmically, with smaller errors being more common than larger ones. 
# The MedAE value of 0.1188 indicates that the median absolute error is higher than the previous model.

# Overall, while the refined random forest model has a lower performance than the previous one, it is still making relatively accurate predictions. 
# Further analysis and testing may be necessary to fully evaluate its performance.

#%%[markdown]

### Refined Random forest Model performance

# Cross validation using sklearn
from sklearn.model_selection import cross_val_score

rf_hyp_scores = cross_val_score(rf_best, 
                         X, y, cv = 5, 
                         scoring='r2')

print("5-fold X-Validation for refined random forest model:")
print("Cross-validation R^2 scores:")
for r_2 in rf_hyp_scores:
    print(round(r_2, 4), end = "  ")
print()
print("Mean R^2:", round(rf_hyp_scores.mean(), 4))


#%%[markdown]

#### The 5-fold cross-validation result for a refined random forest model is displayed in the output. 

# The cross-validation mean R-squared value of 0.2835 indicates that the refined random forest model is 
# not performing as well as the original model, which has an R-squared value of 0.5119. 
# This suggests that the refined random forest model may be overfitting the training data and not generalizing 
# well to new data. 
# The negative R-squared value in one of the folds also suggests that the model is performing worse than a model 
# that simply predicts the mean value of the target variable. 
# 
# Further refinement or another machine learning algorithm may be necessary for further analysis.


#%%[markdown]

### Model - 3
## Gradient Boosting algorithm (Ensemble method)

# Instantiate rf
gb_model = GradientBoostingRegressor(n_estimators = 25, 
                                     learning_rate = 0.1, 
                                     random_state = 2)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)


r2_gb = r2_score(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
msle_gb = mean_squared_log_error(y_test, y_pred_gb)
medae_gb = median_absolute_error(y_test, y_pred_gb)

# Gradient Boosting model evaluation metrics
print("Gradient Boosting model evaluation metrics: ")
print(f"R^2: {r2_gb} ({round(r2_gb * 100, 1)}%)")
print(f"MSE: {mse_gb} ({round(mse_gb * 100, 1)}%)")
print(f"MAE: {mae_gb} ({round(mae_gb * 100, 1)}%)")
print(f"MSLE: {msle_gb} ({round(msle_gb * 100, 1)}%)")
print(f"MedAE: {medae_gb} ({round(medae_gb * 100, 1)}%)")

# Plot the feature importances
importances = pd.Series(data = gb_model.feature_importances_, 
                        index = X.columns)

importances_sorted = importances.sort_values()
importances_sorted.plot(kind = 'barh', color = 'lightgreen')
plt.title('Features Importances (GB)')
plt.show()

#%% [markdown] Interpreting the results of the Gradient Boosting model:

# The gradient boosting model has a R^2 value of 0.6732, 
#       which means that it can explain 67.3% of the variance in the target variable. 
# 
# The MSE (mean squared error) value is very low at 0.019, indicating that the model's predictions 
#      are close to the actual values. 
# 
# The MAE (mean absolute error) value is also low at 0.1138, which means that the average difference between the predicted and actual values is small. 
# 
# The MSLE (mean squared logarithmic error) value is low at 0.0106, which indicates that the model is making accurate predictions across the entire range of target values. 
# 
# The MedAE (median absolute error) value is also low at 0.0952, which means that half of the absolute errors are smaller than this value. 


#%%[markdown]

### Gradient Boosting Model performance
# Cross validation using sklearn
from sklearn.model_selection import cross_val_score

gb_scores = cross_val_score(gb_model, 
                         X, y, cv = 5, 
                         scoring='r2')

print("5-fold X-Validation for Gradient Boosting model:")
print("Cross-validation R^2 scores:")
for r_2 in gb_scores:
    print(round(r_2, 4), end = "  ")
print()
print("Mean R^2:", round(gb_scores.mean(), 4))

#%%[markdown]

# The R-squared value of the cross-validation for the gradient boosting model is 0.369, 
#       which is lower than the original R-squared value of 0.6732. 
# 
# This indicates that the model is overfitting to the training data and is not able to 
#       generalize well to new data. 
# 
# The cross-validation score gives an estimate of how well the model is likely to perform on new data. 
# 
# Since the cross-validation score is lower than the original R-squared value, 
#           it suggests that the model needs to be improved in order to better generalize to new data.


#%%[markdown]
# Overall, these results suggest that the random forest model is a good fit over regression and gradient boosting models 
#     for the data and is making accurate predictions with R^2 (R-squared) value of 93%.

#%% [markdown] 
### Compare the results of the regression tree and random forest models
#  To compare the performance of the regression tree, gradient boosting and random forest models, 
# we can look at the evaluation metrics such as: 
# *  R-squared (R^2), 
# *  mean squared error (MSE), 
# *  mean absolute error (MAE), 
# *  mean squared logarithmic error (MSLE), and 
# *  median absolute error (MedAE). 

# - We can also compare the feature importances of the models.
# From the evaluation metrics, we can see that the random forest model outperforms the regression tree model and gradient boosting model on all metrics. 
# The R^2 value of the random forest model is 0.9304, which is much higher than the R^2 value of the regression tree model (0.4918). 
# The MSE, MAE, MSLE, and MedAE values of the random forest model are also lower than those of the regression tree and gradient boosting models, 
#       indicating that the random forest model is making more accurate predictions.

# We can also compare the feature importances of the two models: 
#     The feature importances of the random forest model are generally higher than those of the regression tree & gradient boosting models, 
#       indicating that the random forest model is able to better capture the relationships between the features and the target variable.

# Overall, the random forest model appears to be a better choice for this dataset as it outperforms the regression tree and gradient boosting model on all evaluation metrics and has higher feature importances

#%% [markdown] 
### Results
# - The random forest model has a high R^2 value of 0.9304, which means that it can explain 93% of the variance in the target variable.
# - The MSE (mean squared error) value is very low at 0.0041, indicating that the model's predictions are very close to the actual values.
# - The MAE (mean absolute error) value is also low at 0.045, which means that the average difference between the predicted and actual values is small.
# - The MSLE (mean squared logarithmic error) value is low at 0.0023, which indicates that the model is making accurate predictions across the entire range of target values.
# - The MedAE (median absolute error) value is also low at 0.03, which means that half of the absolute errors are smaller than this value.
# - Overall, these results suggest that the random forest model is a good fit for this data among all other models and is making accurate predictions.

#%%[markdown]
### Limitations:
# * Due to less number of observations, all the machine learning models are over-fitting the data even with refined and hyperparameter techniques.


#%%[markdown]
############################################################################################################################################################################
##### SUMMARY #####
########################################################################################################################################################################

models = ['Decision Tree', 
          'Random forest', 
          'Refined Random forest', 
          'Gradient boosting']

# R-squared (R^2)
r2 = [round(r2_dt, 2), 
      round(r2_rf, 2), 
      round(r2_rf_best, 2), 
      round(r2_gb, 2)]

# Root mean squared error (RMSE)
rmse = [round(np.sqrt(mse_dt), 2), 
        round(np.sqrt(mse_rf), 2), 
        round(np.sqrt(mse_rf_best), 2), 
        round(np.sqrt(mse_gb), 2)
        ]

# mean absolute error (MAE) 
mae = [round(mae_dt, 2), round(mae_rf, 2), 
       round(mae_rf_best, 2), round(mae_gb, 2)]

# mean squared logarithmic error (MSLE)
msle = [round(msle_dt, 2), 
        round(msle_rf, 2), 
        round(msle_rf_best, 2), 
        round(msle_gb, 2)]

# median absolute error (MedAE). 
medae = [round(medae_dt, 2), 
         round(medae_rf, 2), 
         round(medae_rf_best, 2), 
         round(medae_gb, 2)
         ]

model_perf = {'Model' : models, 
              'R-Squared' : r2, 
              'RMSE' : rmse,
              'MAE' : mae,
              'MSLE' : msle,
              'MedAE' : medae
              }

df_perf = pd.DataFrame(model_perf)
df_perf

# %%
