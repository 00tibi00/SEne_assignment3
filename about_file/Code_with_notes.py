# -*- coding: utf-8 -*-
#Project 3
#Code by Katharina Zimmer

#estimation of the CO2 emissions of electricity of a country for the next day
#country: Denkmark

#Step 1: Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#yearly energy data
raw_den1=pd.read_csv('owid-energy-data.csv')
#source:https://github.com/owid/energy-data?tab=readme-ov-file
#only specific co2 emissions per kWh available
raw_den_co2_hourly=pd.read_csv('DeclarationEmissionHour.csv', sep=";")
#source: https://www.energidataservice.dk/tso-electricity/DeclarationEmissionHour

#clean raw_den1
print(raw_den1)

denmark_data = raw_den1.loc[raw_den1['country'] == 'Denmark']
print(denmark_data)
#show names of columns
print(denmark_data.columns)

drop_columns = [
    'country', 'iso_code', 'population', 'gdp',
    'biofuel_cons_per_capita', 'coal_cons_per_capita',
    'energy_per_capita', 'fossil_elec_per_capita', 'fossil_energy_per_capita',
    'gas_elec_per_capita', 'gas_energy_per_capita', 'hydro_elec_per_capita',
    'hydro_energy_per_capita', 'low_carbon_elec_per_capita',
    'low_carbon_energy_per_capita', 'nuclear_elec_per_capita',
    'nuclear_energy_per_capita', 'oil_elec_per_capita', 'oil_energy_per_capita',
    'other_renewables_elec_per_capita', 'other_renewables_elec_per_capita_exc_biofuel',
    'other_renewables_energy_per_capita', 'per_capita_electricity',
    'renewables_elec_per_capita', 'renewables_energy_per_capita',
    'solar_elec_per_capita', 'solar_energy_per_capita',
    'wind_elec_per_capita', 'wind_energy_per_capita'
]
denmark_data= denmark_data.drop(columns=drop_columns)

print(denmark_data)
print(denmark_data.columns)

columns = ['biofuel_cons_change_pct', 'biofuel_cons_change_twh',
           'biofuel_consumption', 'biofuel_elec_per_capita', 'biofuel_electricity',
           'biofuel_share_elec', 'biofuel_share_energy', 'carbon_intensity_elec',
           'coal_cons_change_pct', 'coal_cons_change_twh', 'coal_consumption',
           'coal_elec_per_capita', 'coal_electricity', 'coal_prod_change_pct',
           'coal_prod_change_twh', 'coal_prod_per_capita', 'coal_production',
           'coal_share_elec', 'coal_share_energy', 'electricity_demand',
           'electricity_generation', 'electricity_share_energy',
           'energy_cons_change_pct', 'energy_cons_change_twh', 'energy_per_gdp',
           'fossil_cons_change_pct', 'fossil_cons_change_twh',
           'fossil_electricity', 'fossil_fuel_consumption', 'fossil_share_elec',
           'fossil_share_energy', 'gas_cons_change_pct', 'gas_cons_change_twh',
           'gas_consumption', 'gas_electricity', 'gas_prod_change_pct',
           'gas_prod_change_twh', 'gas_prod_per_capita', 'gas_production',
           'gas_share_elec', 'gas_share_energy', 'greenhouse_gas_emissions',
           'hydro_cons_change_pct', 'hydro_cons_change_twh', 'hydro_consumption',
           'hydro_electricity', 'hydro_share_elec', 'hydro_share_energy',
           'low_carbon_cons_change_pct', 'low_carbon_cons_change_twh',
           'low_carbon_consumption', 'low_carbon_electricity',
           'low_carbon_share_elec', 'low_carbon_share_energy', 'net_elec_imports',
           'net_elec_imports_share_demand', 'nuclear_cons_change_pct',
           'nuclear_cons_change_twh', 'nuclear_consumption', 'nuclear_electricity',
           'nuclear_share_elec', 'nuclear_share_energy', 'oil_cons_change_pct',
           'oil_cons_change_twh', 'oil_consumption', 'oil_electricity',
           'oil_prod_change_pct', 'oil_prod_change_twh', 'oil_prod_per_capita',
           'oil_production', 'oil_share_elec', 'oil_share_energy',
           'other_renewable_consumption', 'other_renewable_electricity',
           'other_renewable_exc_biofuel_electricity',
           'other_renewables_cons_change_pct', 'other_renewables_cons_change_twh',
           'other_renewables_share_elec',
           'other_renewables_share_elec_exc_biofuel',
           'other_renewables_share_energy', 'primary_energy_consumption',
           'renewables_cons_change_pct', 'renewables_cons_change_twh',
           'renewables_consumption', 'renewables_electricity',
           'renewables_share_elec', 'renewables_share_energy',
           'solar_cons_change_pct', 'solar_cons_change_twh', 'solar_consumption',
           'solar_electricity', 'solar_share_elec', 'solar_share_energy',
           'wind_cons_change_pct', 'wind_cons_change_twh', 'wind_consumption',
           'wind_electricity', 'wind_share_elec', 'wind_share_energy']

# Spalten behalten, die "consumption" oder "electricity" enthalten
keep_columns = [col for col in columns if 'consumption' in col or 'electricity' in col or'carbon' in col]

# Spalten entfernen, die nicht "consumption" oder "electricity" enthalten
drop_columns = [col for col in columns if col not in keep_columns]

#print(drop_columns)

denmark_data= denmark_data.drop(columns=drop_columns)

print(denmark_data)
print(denmark_data.columns)

denmark_data_cleaned = denmark_data.dropna()
#print(denmark_data_cleaned)

#denmark_data_cleaned['carbon_intensity_elec'].plot()

denmark_data_yearly=denmark_data_cleaned

print(denmark_data_cleaned.columns)

drop_columns =['biofuel_consumption', 
       'carbon_intensity_elec', 'coal_consumption',
       'electricity_demand', 'electricity_generation',
       'electricity_share_energy',
       'fossil_fuel_consumption', 'gas_consumption', 
       'hydro_consumption','low_carbon_cons_change_pct',
       'low_carbon_cons_change_twh', 'low_carbon_consumption','low_carbon_electricity',
       'low_carbon_share_elec',
       'low_carbon_share_energy', 'nuclear_consumption',
       'oil_consumption', 'other_renewable_consumption',
       'primary_energy_consumption',
       'renewables_consumption',  'solar_consumption', 'wind_consumption',
       'fossil_electricity','renewables_electricity','other_renewable_exc_biofuel_electricity'
       ,'nuclear_electricity']
       
denmark_data_cleaned= denmark_data_cleaned.drop(columns=drop_columns)
#print(denmark_data_cleaned)

denmark_data_cleaned = denmark_data_cleaned[denmark_data_cleaned['year'].between(2017, 2022)]

print(denmark_data_cleaned)
print(denmark_data_cleaned.columns)
#denmark_data_cleaned['nuclear_electricity'].plot()
#denmark_data_cleaned['hydro_electricity'].plot()


# Data for 2022
new_column_order = ['year', 'biofuel_electricity', 'coal_electricity', 'gas_electricity',
       'oil_electricity', 'other_renewable_electricity',
       'solar_electricity', 'wind_electricity','hydro_electricity']

# Reorder the columns in the DataFrame
denmark_data_cleaned = denmark_data_cleaned[new_column_order]

# Set 'year' as index
denmark_data_cleaned.set_index('year', inplace=True)

# Data for 2022
data_2022 = denmark_data_cleaned.loc[2022]


# Labels for the legend
labels = ['Biofuel Electricity',
    'Coal Electricity',
    'Gas Electricity',
    'Oil Electricity',
    'Other Renewable Electricity',
    'Solar Electricity',
    'Wind Electricity',
    'Hydro Electricity'
]

# Values for the pie chart
sizes = data_2022.values

# Specify your own colors
custom_colors = ['lightgreen', 'black', 'grey', 'orange', 'magenta',
                 'gold', 'lightblue', 'blue']

# Create the pie chart
plt.figure(figsize=(8, 8))
patches, _, _ = plt.pie(sizes, colors=custom_colors, autopct='%1.2f%%', startangle=140, textprops={'fontsize': 14}, pctdistance=1.2)
plt.axis('equal')
plt.title('Electricity Generation in Denmark (2022)',fontsize=14)

# Create legend
plt.legend(patches, labels, loc="lower center", fontsize=12, bbox_to_anchor=(0.5, -0.2), ncol=3)

plt.show()

#clean raw_den_co2_hourly

print(raw_den_co2_hourly)

#drop unnecessary columns
raw_den_co2_hourly = raw_den_co2_hourly.drop(columns=['HourDK', 'FuelAllocationMethod', 'Edition', 
                                            'CO2originPerkWh', 'SO2PerkWh', 'NOxPerkWh', 'NMvocPerkWh', 'COPerkWh', 
                                            'ParticlesPerkWh', 'CoalFlyAshPerkWh', 'CoalSlagPerkWh', 'DesulpPerkWh', 
                                            'FuelGasWastePerkWh', 'BioashPerkWh', 'WasteSlagPerkWh', 
                                            'RadioactiveWastePerkWh'])

#print(raw_den_co2_hourly)

#rename column
raw_den_co2_hourly = raw_den_co2_hourly.rename(columns={'HourUTC': 'Date'})

print(raw_den_co2_hourly.columns)

#date as index
raw_den_co2_hourly = raw_den_co2_hourly.sort_values(by='Date').reset_index(drop=True)

print(raw_den_co2_hourly)

#replace , with . 
raw_den_co2_hourly['CO2PerkWh'] = raw_den_co2_hourly['CO2PerkWh'].str.replace(',', '.').astype(float)
raw_den_co2_hourly['CH4PerkWh'] = raw_den_co2_hourly['CH4PerkWh'].str.replace(',', '.').astype(float)
raw_den_co2_hourly['N2OPerkWh'] = raw_den_co2_hourly['N2OPerkWh'].str.replace(',', '.').astype(float)

#print(raw_den_co2_hourly)

raw_den_co2_hourly.plot()

#create columns for each price area

df_dk1 = raw_den_co2_hourly[raw_den_co2_hourly['PriceArea'] == 'DK1'].reset_index(drop=True)
df_dk2 = raw_den_co2_hourly[raw_den_co2_hourly['PriceArea'] == 'DK2'].reset_index(drop=True)

# rename columns for each price area
df_dk1.rename(columns={'CO2PerkWh': 'CO2PerkWh_DK1', 'CH4PerkWh': 'CH4PerkWh_DK1', 'N2OPerkWh': 'N2OPerkWh_DK1'}, inplace=True)
df_dk2.rename(columns={'CO2PerkWh': 'CO2PerkWh_DK2', 'CH4PerkWh': 'CH4PerkWh_DK2', 'N2OPerkWh': 'N2OPerkWh_DK2'}, inplace=True)

# merge dataframes
df_merged_hourly_co2 = pd.concat([df_dk1, df_dk2], axis=1)

df_merged_hourly_co2 = df_merged_hourly_co2.drop(columns=['PriceArea'])

#print(df_merged_hourly_co2)

# create new index
new_index = pd.date_range(start='2016-12-31 23:00', end='2022-12-31 22:00', freq='H')

# delete old date column
if 'Date' in df_merged_hourly_co2.columns:
    del df_merged_hourly_co2['Date']

# create new column 'Date'
df_merged_hourly_co2['Date'] = new_index

# date as index
df_merged_hourly_co2.set_index('Date', inplace=True)

print(df_merged_hourly_co2)

# create daily data
df_daily_sum = df_merged_hourly_co2.resample('D').sum()

# reset index
df_daily_sum.reset_index(inplace=True)

print(df_daily_sum)
df_daily_sum.info()

#weather data

#data collecction
#source 2017: https://meteostat.net/de/place/dk/copenhagen?s=06180&t=2017-01-01/2017-12-31
#other csv files from same website for different years
raw_meteo_2017=pd.read_csv('daily_copenhagen_2017.csv')
raw_meteo_2018=pd.read_csv('daily_copenhagen_2018.csv')
raw_meteo_2019=pd.read_csv('daily_copenhagen_2019.csv')
raw_meteo_2020=pd.read_csv('daily_copenhagen_2020.csv')
raw_meteo_2021=pd.read_csv('daily_copenhagen_2021.csv')
raw_meteo_2022=pd.read_csv('daily_copenhagen_2022.csv')

print(raw_meteo_2017)

dfs = [raw_meteo_2017, raw_meteo_2018, raw_meteo_2019, raw_meteo_2020, raw_meteo_2021,raw_meteo_2022]

# connect weather data
weather_data = pd.concat(dfs, ignore_index=True)

# display new dataframe
print(weather_data)

#drop columns without sufficient data or without relevance
weather_data = weather_data.drop(columns=['prcp','snow','wpgt','tsun'])

print(weather_data.columns)

print(weather_data.dtypes)

#observe null-values
weather_data.info()

#clean weather data

# show zero values
nan_values = weather_data[weather_data.isna().any(axis=1)]

print(nan_values)

#decision to interpolate nan values because they are only a few values

weather_data_interpolated = weather_data.interpolate()

#print(weather_data_interpolated)

#check if there are null-values left
weather_data_interpolated.info()

#check if interpolation worked
print(weather_data_interpolated.iloc[80:96])

weather_data_interpolated = weather_data_interpolated.rename(columns={'date': 'Date'})

#merge weather data and co2 emissison data
weather_data_interpolated['Date'] = pd.to_datetime(weather_data_interpolated['Date'])
df_daily_sum['Date'] = pd.to_datetime(df_daily_sum['Date'])
merged_df = pd.merge(weather_data_interpolated, df_daily_sum, on='Date')

#show merged data
#print(merged_df)

merged_df.info()

#Eplore data

#show lowest values
df_sort_co2_dk1 = merged_df.sort_values(by = 'CO2PerkWh_DK1', ascending = True)
print(df_sort_co2_dk1['CO2PerkWh_DK1'] [:9] )

#show highest values
df_sort_co2_dk1 = merged_df.sort_values(by = 'CO2PerkWh_DK1', ascending = False)
print(df_sort_co2_dk1['CO2PerkWh_DK1'] [:9])

#show lowest values
df_sort_tavg = merged_df.sort_values(by = 'tavg', ascending = True)
print(df_sort_tavg['tavg'] [:9] )

#show highest values
df_sort_tavg = merged_df.sort_values(by = 'tavg', ascending = False)
print(df_sort_tavg['tavg'] [:9])

#show lowest values
df_sort_ch4_dk1 = merged_df.sort_values(by = 'CH4PerkWh_DK1', ascending = True)
print(df_sort_ch4_dk1['CH4PerkWh_DK1'] [:9] )

df_sort_ch4_dk2 = merged_df.sort_values(by = 'CH4PerkWh_DK2', ascending = True)
print(df_sort_ch4_dk2['CH4PerkWh_DK2'] [:9] )

merged_df.plot(y=['CH4PerkWh_DK2','CH4PerkWh_DK1'])

#no outliers removal because it's hard to tell if there are outliers

#feature selection
df_data=merged_df

#create column with emission data of the day before
df_data['CO2_DK1-1']=df_data['CO2PerkWh_DK1'].shift(1) 
df_data['CO2_DK2-1']=df_data['CO2PerkWh_DK2'].shift(1) 
df_data=df_data.dropna()
print(df_data.head())

print(df_data.columns)

#create 2 dataframes, one for each price area: dk1 and dk2

df_dk1 = df_data[['Date', 'tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres',
                  'CO2PerkWh_DK1', 'CO2_DK1-1']]
#print(df_dk1)

#create a new column with day of the week
df_dk1['Date'] = pd.to_datetime(df_dk1['Date'])

df_dk1['Day'] = df_dk1['Date'].dt.dayofweek

print(df_dk1.columns)

df_dk1=df_dk1.set_index('Date', drop = True ) # set date as index

#change position of columns
df_dk1=df_dk1.iloc[:, [6,7,0,4,1,2,3,5,8]]
print(df_dk1.columns)

#define inputs and outputs
Z_1=df_dk1.values

Y_1=Z_1[:,0]
X_1=Z_1[:,[1,2,3,4,5,6,7,8]] 
#print(Y_1)
#print(X_1)

df_dk2 = df_data[['Date', 'tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres',
                  'CO2PerkWh_DK2', 'CO2_DK2-1']]
print(df_dk2.columns)

df_dk2=df_dk2.set_index('Date', drop = True ) # set date as index

#change position of columns
df_dk2=df_dk2.iloc[:, [6,7,0,4,1,2,3,5]]
print(df_dk2.columns)

#define inputs and outputs
Z_2=df_dk2.values

Y_2=Z_2[:,0]
X_2=Z_2[:,[1,2,3,4,5,6,7]] 
#print(Y_2)
#print(X_2)

#feature selection
#feature selection is done for dk1, it is assumed that the best features for both dataframes are simmilar
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)

#filter methods dk1

#mutual info regression
features = SelectKBest(score_func=mutual_info_regression, k=5) 
fit = features.fit(X_1, Y_1)

# print scores
print("Scores:")
print(fit.scores_)

# transform feature results
features_results = fit.transform(X_1)
#print("feature results:")
#print(features_results)

#f_regression
features = SelectKBest(score_func=f_regression, k=5) 
fit = features.fit(X_1, Y_1) 

# print scores
print("Scores:")
print(fit.scores_)

# transform feature results
features_results = fit.transform(X_1)
#print("feature results:")
#print(features_results)

#wrapper methods

# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# do RFE for up to 4 features because more features would lead to more time for running the models
model=LinearRegression() # LinearRegression Model as Estimator
rfe1=RFE(model,n_features_to_select=1)# using 1 features
rfe2=RFE(model,n_features_to_select=2) # using 2 features
rfe3=RFE(model,n_features_to_select=3)# using 3 features
rfe4=RFE(model,n_features_to_select=4)# using 4 features
fit1=rfe1.fit(X_1,Y_1)
fit2=rfe2.fit(X_1,Y_1)
fit3=rfe3.fit(X_1,Y_1)
fit4=rfe4.fit(X_1,Y_1)

print( "Feature Ranking (Linear Model, 1 features): %s" % (fit1.ranking_)) # wind direction
print( "Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_)) # tmax, co2-1
print( "Feature Ranking (Linear Model, 3 features): %s" % (fit3.ranking_)) # tmin, co2-1
print( "Feature Ranking (Linear Model, 4 features): %s" % (fit4.ranking_)) # windspeed, co2-1

#ensemble methods
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_1, Y_1)
# identify the best feature for forecast
print(model.feature_importances_) # co2-1, windspeed, tmax

#feature engineering 
#both dataframes

# log for temperature
days = df_data['Date'].dt.dayofweek.values

# sinus for weekday
amp = 1 / 7
df_dk1['Sin_day'] = 10 * np.sin(2 * np.pi * amp * days)
df_dk2['Sin_day'] = 10 * np.sin(2 * np.pi * amp * days)


#print(df_dk1.columns)
#print(df_dk2.columns)

#data 2022 is testing data for the model

df_dk1_test = df_dk1[df_dk1.index >= '2022-01-01']
df_dk1 = df_dk1[df_dk1.index < '2022-01-01']

df_dk2_test = df_dk2[df_dk2.index >= '2022-01-01']
df_dk2 = df_dk2[df_dk2.index < '2022-01-01']

#define inputs and outputs
Z_1=df_dk1.values

Y_1=Z_1[:,0]
X_1=Z_1[:,[1,2,3,4,5,6,7,8,9]] 
#print(Y_1)
#print(X_1)

#observe feature importances for dk1
model = RandomForestRegressor()
model.fit(X_1, Y_1)
# identify the best feature for forecast
print(model.feature_importances_)

#define inputs and outputs
Z_2=df_dk1.values

Y_2=Z_2[:,0]
X_2=Z_2[:,[1,2,3,4,5,6,7,8]] 
#print(Y_2)
#print(X_2)

#observe feature importances for dk2
model = RandomForestRegressor()
model.fit(X_2, Y_2)
# identify the best feature for forecast
print(model.feature_importances_)

#same features for both dataframes
#different features could be selected for the price ares
df_dk1_features=df_dk1[['CO2PerkWh_DK1','CO2_DK1-1', 'wspd','tmax','wdir','Sin_day','tmin']]
df_dk2_features=df_dk2[['CO2PerkWh_DK2','CO2_DK2-1', 'wspd','tmax','wdir','Sin_day','tmin']]

#show dataframes
#print(df_dk1_features)
#print(df_dk2_features)


#regression for dk1
from sklearn.model_selection import train_test_split
from sklearn import  metrics
import statsmodels.api as sm

#time series analysis
freq = 1

res = sm.tsa.seasonal_decompose(df_dk1_features['CO2PerkWh_DK1'],
                                period=freq,
                                model='multiplicative')
#model='additive'
resplot = res.plot()

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_dk1_features['CO2PerkWh_DK1'])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df_dk1_features['CO2PerkWh_DK1'])
plt.show()

from pandas.plotting import lag_plot
lag_plot(df_dk1_features['CO2PerkWh_DK1'])
plt.show()

 # split training and test data randomly
 #dk1
Z_1=df_dk1_features.values
#Identify output Y
Y_1=Z_1[:,0]
#Identify input X
X_1=Z_1[:,[1,2,3,4,5,6]]

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1,Y_1)
#print(X_1_train)
#print(y_1_train)

#dk2
Z_2=df_dk2_features.values
#Identify output Y
Y_2=Z_2[:,0]
#Identify input X
X_2=Z_2[:,[1,2,3,4,5,6]]

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2,Y_2)
#print(X_2_train)
#print(y_2_train)

#there is a trained model for dk1 and dk1
#at the moment the dataframes are using the same features
#theoretically it would be possible to do different feature selection but I think it doesn't make sense

#training models

#defining errors

#all models besides SVR
from tabulate import tabulate
def print_errors(y_test, y_pred_REGR, name_REGRmodel):
    MAE = metrics.mean_absolute_error(y_test,y_pred_REGR) 
    MBE = np.mean(y_test- y_pred_REGR) #here we calculate MBE
    MSE = metrics.mean_squared_error(y_test,y_pred_REGR)  
    RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_REGR))
    cvRMSE = RMSE/np.mean(y_test)
    NMBE = MBE/np.mean(y_test)
    
    errors_REGR = [["Mean Absolute Error (MAE)", MAE.round(3)], ["Mean Bias Error (MBE)", MBE.round(3)],
                   ["Mean Squared Error (MSE)", MSE.round(3)], ["Root Mean Squared Error (RMSE)", RMSE.round(3)],
                   ["Coefficient of Variation of RMSE (cvRMSE)", cvRMSE.round(3)], 
                   ["Normalized Mean Bias Error (NMBE)", NMBE.round(3)]]

    return print(tabulate(errors_REGR, headers=["ERROR CALCULATIONS " + name_REGRmodel, "Values"], tablefmt="simplegrid"))

#just SVR
def print_errors_SVR(y_test,y_test_SVR, y_pred_REGR, name_REGRmodel):
    MAE = metrics.mean_absolute_error(y_test_SVR,y_pred_REGR) 
    MBE = np.mean(y_test- y_pred_REGR) #here we calculate MBE
    MSE = metrics.mean_squared_error(y_test_SVR,y_pred_REGR)  
    RMSE = np.sqrt(metrics.mean_squared_error(y_test_SVR,y_pred_REGR))
    cvRMSE = RMSE/np.mean(y_test)
    NMBE = MBE/np.mean(y_test)
    
    errors_REGR = [["Mean Absolute Error (MAE)", MAE.round(3)], ["Mean Bias Error (MBE)", MBE.round(3)],
                   ["Mean Squared Error (MSE)", MSE.round(3)], ["Root Mean Squared Error (RMSE)", RMSE.round(3)],
                   ["Coefficient of Variation of RMSE (cvRMSE)", cvRMSE.round(3)], 
                   ["Normalized Mean Bias Error (NMBE)", NMBE.round(3)]]

    return print(tabulate(errors_REGR, headers=["ERROR CALCULATIONS " + name_REGRmodel, "Values"], tablefmt="simplegrid"))


#linear regression 
from sklearn import  linear_model

#dk1
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_1_train,y_1_train)

# Make predictions using the testing set
y_1_pred_LR = regr.predict(X_1_test)

#plot
plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_1_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_1_pred_LR)
plt.show()

#Evaluate errors
print_errors(y_1_test, y_1_pred_LR, "LINEAR REGRESSION DK1")
#cvRMSE too high

#dk2
# Create linear regression object
regr_2 = linear_model.LinearRegression()

# Train the model using the training sets
regr_2.fit(X_2_train,y_2_train)

# Make predictions using the testing set
y_2_pred_LR = regr_2.predict(X_2_test)

#plot
plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_2_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_2_pred_LR)
plt.show()

#Evaluate errors
print_errors(y_2_test, y_2_pred_LR, "LINEAR REGRESSION DK2")


#support vector regressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#dk1

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train_ss = ss_X.fit_transform(X_1_train)
y_train_ss = ss_y.fit_transform(y_1_train.reshape(-1,1))

regr = SVR(kernel='linear')
#kernel='linear'
#kernel='sigmoid'
#kernel='rbf'

regr.fit(X_train_ss,y_train_ss)

y_pred_SVR = regr.predict(ss_X.fit_transform(X_1_test))
y_test_SVR=ss_y.fit_transform(y_1_test.reshape(-1,1))#It is just one column we have to reshape, otherwise its a line
y_pred_SVR2=ss_y.inverse_transform(y_pred_SVR.reshape(-1,1))

plt.figure()  
plt.plot(y_test_SVR[1:200])
plt.plot(y_pred_SVR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test, y_pred_SVR2)
plt.show()

print_errors_SVR(y_1_test,y_test_SVR, y_pred_SVR, "SUPORT VECTOR REGRESSOR DK1")
#NMBE too high

#dk2
ss_X_2 = StandardScaler()
ss_y_2 = StandardScaler()
X_2_train_ss = ss_X_2.fit_transform(X_2_train)
y_2_train_ss = ss_y_2.fit_transform(y_2_train.reshape(-1,1))

regr_2 = SVR(kernel='linear')
#kernel='linear'
#kernel='sigmoid'
#kernel='rbf'
regr_2.fit(X_2_train_ss, y_2_train_ss)


y_pred_SVR_2 = regr_2.predict(ss_X_2.transform(X_2_test))
y_test_SVR_2 = ss_y_2.fit_transform(y_2_test.reshape(-1,1)) 
y_pred_SVR2_2 = ss_y_2.inverse_transform(y_pred_SVR_2.reshape(-1,1))


plt.figure()  
plt.plot(y_test_SVR_2[1:200])
plt.plot(y_pred_SVR_2[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test, y_pred_SVR2_2)
plt.show()

# no big outliers on scatter plot
print_errors_SVR(y_2_test,y_test_SVR_2, y_pred_SVR_2, "SUPORT VECTOR REGRESSOR DK2")

#decision tree

from sklearn.tree import DecisionTreeRegressor

#dk1
# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor(min_samples_leaf=5)

# Train the model using the training sets
DT_regr_model.fit(X_1_train, y_1_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_1_test)
plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_pred_DT)
plt.show()

#Evaluate errors
print_errors(y_1_test, y_pred_DT, "DECISION TREE DK1")

#dk2
# Create Regression Decision Tree object
DT_regr_model_2 = DecisionTreeRegressor(min_samples_leaf=5)

# Train the model using the training sets
DT_regr_model_2.fit(X_2_train, y_2_train)

# Make predictions using the testing set
y_pred_DT_2 = DT_regr_model.predict(X_2_test)
plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_pred_DT_2[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_pred_DT_2)
plt.show()

#Evaluate errors
print_errors(y_2_test, y_pred_DT_2, "DECISION TREE DK2")

#random forest

from sklearn.ensemble import RandomForestRegressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 4,
              'n_estimators': 300, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}

#dk1
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_1_train, y_1_train)
y_pred_RF = RF_model.predict(X_1_test)

plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_pred_RF)
plt.show()

#Evaluate 
print_errors(y_1_test, y_pred_RF, "RANDOM FOREST DK1")

#dk2
RF_model_2 = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model_2.fit(X_2_train, y_2_train)
y_pred_RF_2 = RF_model_2.predict(X_2_test)

plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_pred_RF_2[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_pred_RF_2)
plt.show()

#Evaluate errors
print_errors(y_2_test, y_pred_RF_2, "RANDOM FOREST DK2")


#test with data from 2022 for every model -> the user can decide which model is the best
#every model is tested below
print("Test data")

#dk1
#create dataframe with features for test data
df_dk1_test=df_dk1_test[['CO2PerkWh_DK1','CO2_DK1-1', 'wspd','tmax','wdir','Sin_day','tmin']]
#print(df_dk1_test)

#identify input and outputs
Z=df_dk1_test.values
Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6]]

#dk2
#create dataframe with features for test data
df_dk2_test=df_dk2_test[['CO2PerkWh_DK2','CO2_DK2-1', 'wspd','tmax','wdir','Sin_day','tmin']]
#print(df_dk2_test)

#identify input and outputs
Z_2 = df_dk2_test.values
Y_2 = Z_2[:,0]
X_2 = Z_2[:,[1,2,3,4,5,6]]

#linear regression
#dk1

# Make predictions using the testing set
y_1_pred_LR = regr.predict(X)

#plot
plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_1_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_1_pred_LR)
plt.show()

#Evaluate errors
print_errors(Y, y_1_pred_LR, "LINEAR REGRESSION DK1")

#dk2

# Make predictions using the testing set
y_2_pred_LR = regr_2.predict(X_2)

#plot
plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_2_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_2_pred_LR)
plt.show()

#Evaluate errors
print_errors(Y_2, y_2_pred_LR, "LINEAR REGRESSION DK2")

#support vector regressor
#dk1
X_test_ss = ss_X.transform(X)
y_test_ss = ss_y.transform(Y.reshape(-1, 1))


y_pred_test_ss = regr.predict(X_test_ss)
y_pred_test = ss_y.inverse_transform(y_pred_test_ss.reshape(-1, 1))

plt.figure()  
plt.plot(y_test_ss[1:200])
plt.plot(y_pred_test_ss[1:200])
plt.show()

plt.figure()  
plt.scatter(Y, y_pred_test)
plt.show()

print_errors_SVR(Y,y_test_ss, y_pred_test_ss, "SUPORT VECTOR REGRESSOR DK1")


#dk2
X_test_ss_2 = ss_X.transform(X_2)
y_test_ss_2 = ss_y.transform(Y_2.reshape(-1, 1))

y_pred_test_ss_2 = regr_2.predict(X_test_ss_2)
y_pred_test_2 = ss_y.inverse_transform(y_pred_test_ss_2.reshape(-1, 1))

plt.figure()  
plt.plot(y_test_ss_2[1:200])
plt.plot(y_pred_test_ss_2[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2, y_pred_test_2)
plt.show()

print_errors_SVR(Y_2,y_test_ss_2, y_pred_test_ss_2, "SUPORT VECTOR REGRESSOR DK2")


#decision tree
#dk1

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X)
plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_pred_DT)
plt.show()

#Evaluate errors
print_errors(Y, y_pred_DT, "DECISION TREE DK1")

#dk2
# Make predictions using the testing set
y_pred_DT_2 = DT_regr_model.predict(X_2)
plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_pred_DT_2[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_pred_DT_2)
plt.show()

#Evaluate errors
print_errors(Y_2, y_pred_DT_2, "DECISION TREE DK2")


#RF_model = RandomForestRegressor()
#dk1
y_pred_RF = RF_model.predict(X)

plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_pred_RF)
plt.show()

#Evaluate errors
print_errors(Y, y_pred_RF, "RANDOM FOREST DK1")

#dk2
y_pred_RF_2 = RF_model_2.predict(X_2)

plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_pred_RF_2[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_pred_RF_2)
plt.show()

#Evaluate errors
print_errors(Y_2, y_pred_RF_2, "RANDOM FOREST DK2")


