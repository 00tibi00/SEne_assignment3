import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics, linear_model
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from dash import Dash, dcc, html, Input, Output, dash_table

from deferror import df_errors, df_errors_SVR

#--------------------------------PIE PLOTS
raw_den1=pd.read_csv('data/owid-energy-data.csv')
raw_den_co2_hourly=pd.read_csv('data/DeclarationEmissionHour.csv', sep=";")
denmark_data = raw_den1.loc[raw_den1['country'] == 'Denmark']
raw_den_hourly =pd.read_csv('data/ConsumptionCoverageNationalDecl.csv', sep=";")

#PIE PLOT1-----------------------------
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
    'wind_elec_per_capita', 'wind_energy_per_capita']
denmark_data= denmark_data.drop(columns=drop_columns)
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

keep_columns = [col for col in columns if 'consumption' in col or 'electricity' in col or'carbon' in col]
drop_columns = [col for col in columns if col not in keep_columns]
denmark_data= denmark_data.drop(columns=drop_columns)
denmark_data_cleaned = denmark_data.dropna()
denmark_data_yearly=denmark_data_cleaned
drop_columns =['biofuel_consumption', 'biofuel_electricity',
       'carbon_intensity_elec', 'coal_consumption', 'coal_electricity',
       'electricity_demand', 'electricity_generation',
       'electricity_share_energy',
       'fossil_fuel_consumption', 'gas_consumption', 'gas_electricity',
       'hydro_consumption', 'hydro_electricity', 'low_carbon_cons_change_pct',
       'low_carbon_cons_change_twh', 'low_carbon_consumption',
       'low_carbon_electricity', 'low_carbon_share_elec',
       'low_carbon_share_energy', 'nuclear_consumption', 'nuclear_electricity',
       'oil_consumption', 'oil_electricity', 'other_renewable_consumption',
       'other_renewable_electricity',
       'other_renewable_exc_biofuel_electricity', 'primary_energy_consumption',
       'renewables_consumption',  'solar_consumption',
       'solar_electricity', 'wind_consumption', 'wind_electricity']

denmark_data_cleaned= denmark_data_cleaned.drop(columns=drop_columns)
denmark_data_cleaned = denmark_data_cleaned[denmark_data_cleaned['year'].between(2017, 2022)]


data_2022 = denmark_data_cleaned[denmark_data_cleaned['year'] == 2022]
labels = ['Fossils', 'Renewables                 ']
sizes = [data_2022['fossil_electricity'].iloc[0], data_2022['renewables_electricity'].iloc[0]]

color2 = ['#FEB139','#16FF00'] 
trace2 = go.Pie(labels=labels, values=sizes, marker=dict(colors=color2), hoverinfo='none')
layout2 = go.Layout()
fig2 = go.Figure(data=[trace2], layout=layout2)

fig2.update_layout(
    margin={"r":50,"t":25,"l":10,"b":25}, 
    font=dict(size=16),
    legend=dict(
        x=0.0,
        y=0.5, 
        traceorder='normal', 
        font = dict(size = 20, family="Raleway"), 
        ),   
    paper_bgcolor='#F0FFF0',
    plot_bgcolor='#F0FFF0'
)

#PIE PLOT2-----------------------------
denmark_data2 = raw_den1.loc[raw_den1['country'] == 'Denmark']
drop_columns2 = [
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
denmark_data2 = denmark_data2.drop(columns=drop_columns2)
columns2 = ['biofuel_cons_change_pct', 'biofuel_cons_change_twh',
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
keep_columns2 = [col for col in columns2 if 'consumption' in col or 'electricity' in col or'carbon' in col]
drop_columns2 = [col for col in columns2 if col not in keep_columns2]
denmark_data2 = denmark_data2.drop(columns=drop_columns2)
denmark_data_cleaned2 = denmark_data2.dropna()
denmark_data_yearly2=denmark_data_cleaned2
drop_columns2 =['biofuel_consumption', 
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
       
denmark_data_cleaned2= denmark_data_cleaned2.drop(columns=drop_columns2)
denmark_data_cleaned2 = denmark_data_cleaned2[denmark_data_cleaned2['year'].between(2017, 2022)]

new_column_order2= ['year', 'biofuel_electricity', 'coal_electricity', 'gas_electricity',
       'oil_electricity', 'other_renewable_electricity',
       'solar_electricity', 'wind_electricity','hydro_electricity']

denmark_data_cleaned2 = denmark_data_cleaned2[new_column_order2]
denmark_data_cleaned2.set_index('year', inplace=True)

data_2022_2 = denmark_data_cleaned2.loc[2022]
labels2 = ['Biofuel Electricity', 'Coal Electricity', 'Gas Electricity', 'Oil Electricity',
    'Other Renewables', 'Solar Electricity', 'Wind Electricity', 'Hydro Electricity']
sizes3 = data_2022_2.values
color3 = ["#FF5733", "#7FFF00", "#FF6347", "#9370DB", "#FF1493", "#32CD32", "#FFA500", "#6A5ACD"]
trace3 = go.Pie(labels=labels2, values=sizes3, marker=dict(colors=color3 ), hoverinfo='none')
layout3 = go.Layout()
fig3 = go.Figure(data=[trace3], layout=layout3)

fig3.update_layout(
    margin={"r":80,"t":25,"l":10,"b":45}, 
    font=dict(size=16),
    legend=dict(
        x= -0.3,
        y=0.5, 
        traceorder='normal', 
        font = dict(size = 20, family="Raleway"), 
        ),
    paper_bgcolor='#F0FFF0',
    plot_bgcolor='#F0FFF0'
)


#REGRESSION MODELS--------------
raw_den1=pd.read_csv('data/owid-energy-data.csv')
raw_den_co2_hourly=pd.read_csv('data/DeclarationEmissionHour.csv', sep=";")

denmark_data = raw_den1.loc[raw_den1['country'] == 'Denmark']
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
denmark_data = denmark_data.drop(columns=drop_columns)

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
           'wind_electricity', 'wind_share_elec', 'wind_share_energy'
]

keep_columns = [col for col in columns if 'consumption' in col or 'electricity' in col or 'carbon' in col]

drop_columns = [col for col in columns if col not in keep_columns]

denmark_data = denmark_data.drop(columns=drop_columns)

denmark_data_cleaned = denmark_data.dropna()

denmark_data_cleaned = denmark_data_cleaned[denmark_data_cleaned['year'].between(2017, 2022)]

new_column_order = ['year', 'biofuel_electricity', 'coal_electricity', 'gas_electricity',
       'oil_electricity', 'other_renewable_electricity',
       'solar_electricity', 'wind_electricity','hydro_electricity']

denmark_data_cleaned = denmark_data_cleaned[new_column_order]

denmark_data_cleaned.set_index('year', inplace=True)

data_2022 = denmark_data_cleaned.loc[2022]

labels = ['Biofuel Electricity',
    'Coal Electricity',
    'Gas Electricity',
    'Oil Electricity',
    'Other Renewable Electricity',
    'Solar Electricity',
    'Wind Electricity',
    'Hydro Electricity'
]

sizes = data_2022.values

custom_colors = ['lightgreen', 'black', 'grey', 'orange', 'magenta',
                 'gold', 'lightblue', 'blue']


raw_den_co2_hourly = raw_den_co2_hourly.drop(columns=['HourDK', 'FuelAllocationMethod', 'Edition', 
                                            'CO2originPerkWh', 'SO2PerkWh', 'NOxPerkWh', 'NMvocPerkWh', 'COPerkWh', 
                                            'ParticlesPerkWh', 'CoalFlyAshPerkWh', 'CoalSlagPerkWh', 'DesulpPerkWh', 
                                            'FuelGasWastePerkWh', 'BioashPerkWh', 'WasteSlagPerkWh', 
                                            'RadioactiveWastePerkWh'])

raw_den_co2_hourly = raw_den_co2_hourly.rename(columns={'HourUTC': 'Date'})

raw_den_co2_hourly = raw_den_co2_hourly.sort_values(by='Date').reset_index(drop=True)

raw_den_co2_hourly['CO2PerkWh'] = raw_den_co2_hourly['CO2PerkWh'].str.replace(',', '.').astype(float)
raw_den_co2_hourly['CH4PerkWh'] = raw_den_co2_hourly['CH4PerkWh'].str.replace(',', '.').astype(float)
raw_den_co2_hourly['N2OPerkWh'] = raw_den_co2_hourly['N2OPerkWh'].str.replace(',', '.').astype(float)

df_dk1 = raw_den_co2_hourly[raw_den_co2_hourly['PriceArea'] == 'DK1'].reset_index(drop=True)
df_dk2 = raw_den_co2_hourly[raw_den_co2_hourly['PriceArea'] == 'DK2'].reset_index(drop=True)

df_dk1.rename(columns={'CO2PerkWh': 'CO2PerkWh_DK1', 'CH4PerkWh': 'CH4PerkWh_DK1', 'N2OPerkWh': 'N2OPerkWh_DK1'}, inplace=True)
df_dk2.rename(columns={'CO2PerkWh': 'CO2PerkWh_DK2', 'CH4PerkWh': 'CH4PerkWh_DK2', 'N2OPerkWh': 'N2OPerkWh_DK2'}, inplace=True)

df_merged_hourly_co2 = pd.concat([df_dk1, df_dk2], axis=1)

df_merged_hourly_co2 = df_merged_hourly_co2.drop(columns=['PriceArea'])

new_index = pd.date_range(start='2016-12-31 23:00', end='2022-12-31 22:00', freq='h')

if 'Date' in df_merged_hourly_co2.columns:
    del df_merged_hourly_co2['Date']

df_merged_hourly_co2['Date'] = new_index

df_merged_hourly_co2.set_index('Date', inplace=True)

df_daily_sum = df_merged_hourly_co2.resample('D').sum()

df_daily_sum.reset_index(inplace=True)

raw_meteo_2017=pd.read_csv('data/daily_copenhagen_2017.csv')
raw_meteo_2018=pd.read_csv('data/daily_copenhagen_2018.csv')
raw_meteo_2019=pd.read_csv('data/daily_copenhagen_2019.csv')
raw_meteo_2020=pd.read_csv('data/daily_copenhagen_2020.csv')
raw_meteo_2021=pd.read_csv('data/daily_copenhagen_2021.csv')
raw_meteo_2022=pd.read_csv('data/daily_copenhagen_2022.csv')

dfs = [raw_meteo_2017, raw_meteo_2018, raw_meteo_2019, raw_meteo_2020, raw_meteo_2021,raw_meteo_2022]


weather_data = pd.concat(dfs, ignore_index=True)

weather_data = weather_data.drop(columns=['prcp','snow','wpgt','tsun'])

nan_values = weather_data[weather_data.isna().any(axis=1)]

weather_data_interpolated = weather_data.interpolate()

weather_data_interpolated = weather_data.interpolate()

weather_data_interpolated.info()

weather_data_interpolated = weather_data_interpolated.rename(columns={'date': 'Date'})

weather_data_interpolated['Date'] = pd.to_datetime(weather_data_interpolated['Date'])
df_daily_sum['Date'] = pd.to_datetime(df_daily_sum['Date'])
merged_df = pd.merge(weather_data_interpolated, df_daily_sum, on='Date')

df_sort_co2_dk1 = merged_df.sort_values(by = 'CO2PerkWh_DK1', ascending = True)
df_sort_co2_dk1 = merged_df.sort_values(by = 'CO2PerkWh_DK1', ascending = False)

df_sort_tavg = merged_df.sort_values(by = 'tavg', ascending = True)
df_sort_tavg = merged_df.sort_values(by = 'tavg', ascending = False)

df_sort_ch4_dk1 = merged_df.sort_values(by = 'CH4PerkWh_DK1', ascending = True)
df_sort_ch4_dk2 = merged_df.sort_values(by = 'CH4PerkWh_DK2', ascending = True)

df_data=merged_df

df_data['CO2_DK1-1']=df_data['CO2PerkWh_DK1'].shift(1) 
df_data['CO2_DK2-1']=df_data['CO2PerkWh_DK2'].shift(1) 
df_data=df_data.dropna()

df_dk1 = df_data[['Date', 'tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'CO2PerkWh_DK1', 'CO2_DK1-1']]

df_dk1['Date'] = pd.to_datetime(df_dk1['Date'])

df_dk1['Day'] = df_dk1['Date'].dt.dayofweek

df_dk1=df_dk1.set_index('Date', drop = True )

df_dk1=df_dk1.iloc[:, [6,7,0,4,1,2,3,5,8]]

Z_1=df_dk1.values

Y_1=Z_1[:,0]
X_1=Z_1[:,[1,2,3,4,5,6,7,8]] 

df_dk2 = df_data[['Date', 'tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres',
                  'CO2PerkWh_DK2', 'CO2_DK2-1']]

df_dk2=df_dk2.set_index('Date', drop = True )

df_dk2=df_dk2.iloc[:, [6,7,0,4,1,2,3,5]]

Z_2=df_dk2.values

Y_2=Z_2[:,0]
X_2=Z_2[:,[1,2,3,4,5,6,7]] 

features = SelectKBest(score_func=mutual_info_regression, k=5) 
fit = features.fit(X_1, Y_1)

print("Scores:")
print(fit.scores_)

features_results = fit.transform(X_1)

features = SelectKBest(score_func=f_regression, k=5) 
fit = features.fit(X_1, Y_1) 

features_results = fit.transform(X_1)

model=LinearRegression()
rfe1=RFE(model,n_features_to_select=1)
rfe2=RFE(model,n_features_to_select=2)
rfe3=RFE(model,n_features_to_select=3)
rfe4=RFE(model,n_features_to_select=4)
fit1=rfe1.fit(X_1,Y_1)
fit2=rfe2.fit(X_1,Y_1)
fit3=rfe3.fit(X_1,Y_1)
fit4=rfe4.fit(X_1,Y_1)

model = RandomForestRegressor()
model.fit(X_1, Y_1)

days = df_data['Date'].dt.dayofweek.values

amp = 1 / 7
df_dk1['Sin_day'] = 10 * np.sin(2 * np.pi * amp * days)
df_dk2['Sin_day'] = 10 * np.sin(2 * np.pi * amp * days)

df_dk1_test = df_dk1[df_dk1.index >= '2022-01-01']
df_dk1 = df_dk1[df_dk1.index < '2022-01-01']

df_dk2_test = df_dk2[df_dk2.index >= '2022-01-01']
df_dk2 = df_dk2[df_dk2.index < '2022-01-01']

Z_1=df_dk1.values

Y_1=Z_1[:,0]
X_1=Z_1[:,[1,2,3,4,5,6,7,8,9]] 

model = RandomForestRegressor()
model.fit(X_1, Y_1)

model = RandomForestRegressor()
model.fit(X_2, Y_2)

df_dk1_features=df_dk1[['CO2PerkWh_DK1','CO2_DK1-1', 'wspd','tmax','wdir','Sin_day','tmin']]
df_dk2_features=df_dk2[['CO2PerkWh_DK2','CO2_DK2-1', 'wspd','tmax','wdir','Sin_day','tmin']]

freq = 1

res = sm.tsa.seasonal_decompose(df_dk1_features['CO2PerkWh_DK1'],
                                period=freq,
                                model='multiplicative')

resplot = res.plot()

autocorrelation_plot(df_dk1_features['CO2PerkWh_DK1'])
plt.show()
plot_acf(df_dk1_features['CO2PerkWh_DK1'])
plt.show()
lag_plot(df_dk1_features['CO2PerkWh_DK1'])
plt.show()


Z_1=df_dk1_features.values
Y_1=Z_1[:,0]
X_1=Z_1[:,[1,2,3,4,5,6]]

X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1,Y_1)

Z_2=df_dk2_features.values
Y_2=Z_2[:,0]
X_2=Z_2[:,[1,2,3,4,5,6]]

X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2,Y_2)

regr = linear_model.LinearRegression()
regr.fit(X_1_train,y_1_train)
y_1_pred_LR = regr.predict(X_1_test)

plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_1_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_1_pred_LR)
plt.show()

RR_ERDK1_LR = df_errors(y_1_test, y_1_pred_LR, "LINEAR REGRESSION DK1")

regr_2 = linear_model.LinearRegression()
regr_2.fit(X_2_train,y_2_train)
y_2_pred_LR = regr_2.predict(X_2_test)

plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_2_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_2_pred_LR)
plt.show()

RR_ERDK2_LR = df_errors(y_2_test, y_2_pred_LR, "LINEAR REGRESSION DK2")

ss_X = StandardScaler()
ss_y = StandardScaler()
X_train_ss = ss_X.fit_transform(X_1_train)
y_train_ss = ss_y.fit_transform(y_1_train.reshape(-1,1))

regr = SVR(kernel='linear')
regr.fit(X_train_ss,y_train_ss)

y_pred_SVR = regr.predict(ss_X.fit_transform(X_1_test))
y_test_SVR= ss_y.fit_transform(y_1_test.reshape(-1,1))
y_pred_SVR2= ss_y.inverse_transform(y_pred_SVR.reshape(-1,1))

plt.figure()  
plt.plot(y_test_SVR[1:200])
plt.plot(y_pred_SVR[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test, y_pred_SVR2)
plt.show()
RR_ERDK1_SV = df_errors_SVR(y_1_test,y_test_SVR, y_pred_SVR, "SUPORT VECTOR REGRESSOR DK1")

ss_X_2 = StandardScaler()
ss_y_2 = StandardScaler()
X_2_train_ss = ss_X_2.fit_transform(X_2_train)
y_2_train_ss = ss_y_2.fit_transform(y_2_train.reshape(-1,1))

regr_2 = SVR(kernel='linear')
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


RR_ERDK2_SV= df_errors_SVR(y_2_test,y_test_SVR_2, y_pred_SVR_2, "SUPORT VECTOR REGRESSOR DK2")

DT_regr_model = DecisionTreeRegressor(min_samples_leaf=5)
DT_regr_model.fit(X_1_train, y_1_train)

y_pred_DT = DT_regr_model.predict(X_1_test)

plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_pred_DT)
plt.show()


RR_ERDK1_DT=df_errors(y_1_test, y_pred_DT, "DECISION TREE DK1")
DT_regr_model_2 = DecisionTreeRegressor(min_samples_leaf=5)

DT_regr_model_2.fit(X_2_train, y_2_train)

y_pred_DT_2 = DT_regr_model.predict(X_2_test)

plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_pred_DT_2[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_pred_DT_2)
plt.show()

RR_ERDK2_DT = df_errors(y_2_test, y_pred_DT_2, "DECISION TREE DK2")


parameters = {'bootstrap': True,
              'min_samples_leaf': 4,
              'n_estimators': 300, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}

RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_1_train, y_1_train)
y_pred_RF = RF_model.predict(X_1_test)

plt.figure()  
plt.plot(y_1_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()

plt.figure()  
plt.scatter(y_1_test,y_pred_RF)
plt.show()

RR_ERDK1_RF = df_errors(y_1_test, y_pred_RF, "RANDOM FOREST DK1")

RF_model_2 = RandomForestRegressor(**parameters)
RF_model_2.fit(X_2_train, y_2_train)
y_pred_RF_2 = RF_model_2.predict(X_2_test)

plt.figure()  
plt.plot(y_2_test[1:200])
plt.plot(y_pred_RF_2[1:200])
plt.show()

plt.figure()  
plt.scatter(y_2_test,y_pred_RF_2)
plt.show()

RR_ERDK2_RF = df_errors(y_2_test, y_pred_RF_2, "RANDOM FOREST DK2")

df_dk1_test=df_dk1_test[['CO2PerkWh_DK1','CO2_DK1-1', 'wspd','tmax','wdir','Sin_day','tmin']]

Z=df_dk1_test.values
Y=Z[:,0]
X=Z[:,[1,2,3,4,5,6]]

df_dk2_test=df_dk2_test[['CO2PerkWh_DK2','CO2_DK2-1', 'wspd','tmax','wdir','Sin_day','tmin']]
Z_2 = df_dk2_test.values
Y_2 = Z_2[:,0]
X_2 = Z_2[:,[1,2,3,4,5,6]]
y_1_pred_LR = regr.predict(X)

plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_1_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_1_pred_LR)
plt.show()

figDK1LR = go.Figure()
figDK1LR.add_trace(go.Scatter(y=Y[1:200], mode='lines', name='Actual'))
figDK1LR.add_trace(go.Scatter(y=y_1_pred_LR[1:200], mode='lines', name='Predicted'))
figDK1LR.update_layout(title='Linear Regression DK1 Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK1LR.show()

figDK1LR_s = go.Figure()
figDK1LR_s.add_trace(go.Scatter(x=Y, y=y_1_pred_LR, mode='markers', name='Linear Regression DK1 Test Data'))
figDK1LR_s.update_layout(title='Linear Regression DK1 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK1LR_s.show()

ERDK1_LR = df_errors(Y, y_1_pred_LR, "LINEAR REGRESSION DK1")

y_2_pred_LR = regr_2.predict(X_2)

plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_2_pred_LR[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_2_pred_LR)
plt.show()

figDK2LR = go.Figure()
figDK2LR.add_trace(go.Scatter(y=Y_2[1:200], mode='lines', name='Actual'))
figDK2LR.add_trace(go.Scatter(y=y_2_pred_LR[1:200], mode='lines', name='Predicted'))
figDK2LR.update_layout(title='Linear Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK2LR.show()

figDK2LR_s = go.Figure()
figDK2LR_s.add_trace(go.Scatter(x=Y_2, y=y_2_pred_LR, mode='markers', name='Linear Regression DK2 Test Data'))
figDK2LR_s.update_layout(title='Linear Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK2LR_s.show()

ERDK2_LR = df_errors(Y_2, y_2_pred_LR, "LINEAR REGRESSION DK2")

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

figDK1SVR = go.Figure()
figDK1SVR.add_trace(go.Scatter(y=y_test_ss.flatten()[1:200], mode='lines', name='Actual'))
figDK1SVR.add_trace(go.Scatter(y=y_pred_test_ss.flatten()[1:200], mode='lines', name='Predicted'))
figDK1SVR.update_layout(title='Support Vector Regression DK1 Test Data - Actual vs Predicted (Scaled)', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK1SVR.show()

figDK1SVR_s = go.Figure()
figDK1SVR_s.add_trace(go.Scatter(x=Y, y=y_pred_test.flatten(), mode='markers', name='Support Vector Regression DK1 Test Data'))
figDK1SVR_s.update_layout(title='Support Vector Regression DK1 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK1SVR_s.show()

ERDK1_SV = df_errors_SVR(Y,y_test_ss, y_pred_test_ss, "SUPORT VECTOR REGRESSOR DK1")

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

figDK2SVR = go.Figure()
figDK2SVR.add_trace(go.Scatter(y=y_test_ss_2.flatten()[1:200], mode='lines', name='Actual'))
figDK2SVR.add_trace(go.Scatter(y=y_pred_test_ss_2.flatten()[1:200], mode='lines', name='Predicted'))
figDK2SVR.update_layout(title='Support Vector Regression DK2 Test Data - Actual vs Predicted (Scaled)', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK2SVR.show()

figDK2SVR_s = go.Figure()
figDK2SVR_s.add_trace(go.Scatter(x=Y_2, y=y_pred_test_2.flatten(), mode='markers', name='Support Vector Regression DK2 Test Data'))
figDK2SVR_s.update_layout(title='Support Vector Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK2SVR_s.show()

ERDK2_SV = df_errors_SVR(Y_2,y_test_ss_2, y_pred_test_ss_2, "SUPORT VECTOR REGRESSOR DK2")

y_pred_DT = DT_regr_model.predict(X)
plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_pred_DT)
plt.show()


figDK1DT = go.Figure()
figDK1DT.add_trace(go.Scatter(y=Y[1:200], mode='lines', name='Actual'))
figDK1DT.add_trace(go.Scatter(y=y_pred_DT[1:200], mode='lines', name='Predicted'))
figDK1DT.update_layout(title='Decision Tree Regression Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK1DT.show()

figDK1DT_s = go.Figure()
figDK1DT_s.add_trace(go.Scatter(x=Y, y=y_pred_DT, mode='markers', name='Decision Tree Regression Test Data'))
figDK1DT_s.update_layout(title='Decision Tree Regression Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK1DT_s.show()

ERDK1_DT = df_errors(Y, y_pred_DT, "DECISION TREE DK1")

y_pred_DT_2 = DT_regr_model.predict(X_2)

plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_pred_DT_2[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_pred_DT_2)
plt.show()


figDK2DT = go.Figure()
figDK2DT.add_trace(go.Scatter(y=Y_2[1:200], mode='lines', name='Actual'))
figDK2DT.add_trace(go.Scatter(y=y_pred_DT_2[1:200], mode='lines', name='Predicted'))
figDK2DT.update_layout(title='Decision Tree Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK2DT.show()

figDK2DT_s = go.Figure()
figDK2DT_s.add_trace(go.Scatter(x=Y_2, y=y_pred_DT_2, mode='markers', name='Decision Tree Regression DK2 Test Data'))
figDK2DT_s.update_layout(title='Decision Tree Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK2DT_s.show()

ERDK2_DT = df_errors(Y_2, y_pred_DT_2, "DECISION TREE DK2")

y_pred_RF = RF_model.predict(X)

plt.figure()  
plt.plot(Y[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()

plt.figure()  
plt.scatter(Y,y_pred_RF)
plt.show()


figDK1RF = go.Figure()
figDK1RF.add_trace(go.Scatter(y=Y[1:200], mode='lines', name='Actual'))
figDK1RF.add_trace(go.Scatter(y=y_pred_RF[1:200], mode='lines', name='Predicted'))
figDK1RF.update_layout(title='Random Forest Regression DK1 Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK1RF.show()

figDK1RF_s = go.Figure()
figDK1RF_s.add_trace(go.Scatter(x=Y, y=y_pred_RF, mode='markers', name='Random Forest Regression DK1 Test Data'))
figDK1RF_s.update_layout(title='Random Forest Regression DK1 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK1RF_s.show()

ERDK1_RF = df_errors(Y, y_pred_RF, "RANDOM FOREST DK1")
y_pred_RF_2 = RF_model_2.predict(X_2)

plt.figure()  
plt.plot(Y_2[1:200])
plt.plot(y_pred_RF_2[1:200])
plt.show()

plt.figure()  
plt.scatter(Y_2,y_pred_RF_2)
plt.show()

figDK2RF = go.Figure()
figDK2RF.add_trace(go.Scatter(y=Y_2[1:200], mode='lines', name='Actual'))
figDK2RF.add_trace(go.Scatter(y=y_pred_RF_2[1:200], mode='lines', name='Predicted'))
figDK2RF.update_layout(title='Random Forest Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Index', yaxis_title='CO2 Per kWh')
figDK2RF.show()

figDK2RF_s = go.Figure()
figDK2RF_s.add_trace(go.Scatter(x=Y_2, y=y_pred_RF_2, mode='markers', name='Random Forest Regression DK2 Test Data'))
figDK2RF_s.update_layout(title='Random Forest Regression DK2 Test Data - Actual vs Predicted', xaxis_title='Actual', yaxis_title='Predicted')
figDK2RF_s.show()

ERDK2_RF = df_errors(Y_2, y_pred_RF_2, "RANDOM FOREST DK2")



ER_DK1 = [ERDK1_LR, ERDK1_SV,ERDK1_RF, ERDK1_DT]
ER_DK2 = [ERDK2_LR, ERDK2_SV,ERDK2_RF, ERDK2_DT]









colorsG = ['#F806CC','#00F5FF']

def updateGRAPH_layout(fig):
    fig.update_layout(
        font=dict(size=32, family="Raleway"),
        legend=dict(
            traceorder='normal', 
            font = dict(size = 32, family="Raleway", color = 'white'), 
            ),                   
        title_font=dict(color='white', family="Raleway"), 
        xaxis=dict(
            tickfont=dict(color='white', family="Raleway"), 
            color='white'
            ),
        yaxis=dict(
            tickfont=dict(color='white', family="Raleway"),   
            color='white',
            tickformat='.2f'
            ),
        xaxis_gridcolor='rgba(255, 255, 255, 0.5)',
        yaxis_gridcolor='rgba(255, 255, 255, 0.5)',  
        paper_bgcolor='#211C6A',
        plot_bgcolor='#211C6A'
        )
    for i, trace in enumerate(fig.data):
        trace.line.width = 3
        trace.marker.color = colorsG[i] 
    return fig

figDK1LR = updateGRAPH_layout(figDK1LR)
figDK1SVR = updateGRAPH_layout(figDK1SVR)
figDK1DT = updateGRAPH_layout(figDK1DT)
figDK1RF = updateGRAPH_layout(figDK1RF)

figDK2LR = updateGRAPH_layout(figDK2LR)
figDK2SVR = updateGRAPH_layout(figDK2SVR)
figDK2DT = updateGRAPH_layout(figDK2DT)
figDK2RF = updateGRAPH_layout(figDK2RF)


colorsG = ['#F0FF42','#FB2576']


def updateGRAPHscatter_layout(fig):
    fig.update_layout(
        font=dict(size=32, family="Raleway"),
        legend=dict( 
            traceorder='normal', 
            font = dict(size = 32, 
                        family="Raleway", 
                        color = 'white'), 
            ),                   
        title_font=dict(color='white', family="Raleway"), 
        xaxis=dict(
            tickfont=dict(color='white', family="Raleway"), 
            color='white'),
        yaxis=dict(
            tickfont=dict(color='white', family="Raleway"),   
            color='white'),
        xaxis_gridcolor='rgba(255, 255, 255, 0.5)',
        yaxis_gridcolor='rgba(255, 255, 255, 0.5)',  
        paper_bgcolor='#211C6A',
        plot_bgcolor='#211C6A',
        )
    for i, trace in enumerate(fig.data):
        trace.marker.color = colorsG[1] 
    return fig


figDK1LR_s = updateGRAPHscatter_layout(figDK1LR_s)
figDK1SVR_s = updateGRAPHscatter_layout(figDK1SVR_s)
figDK1DT_s = updateGRAPHscatter_layout(figDK1DT_s)
figDK1RF_s = updateGRAPHscatter_layout(figDK1RF_s)

figDK2LR_s = updateGRAPHscatter_layout(figDK2LR_s)
figDK2SVR_s = updateGRAPHscatter_layout(figDK2SVR_s)
figDK2DT_s = updateGRAPHscatter_layout(figDK2DT_s)
figDK2RF_s = updateGRAPHscatter_layout(figDK2RF_s)

#DK INTERACTIVE MAP-------------------------------------------
geojson_file = "data/dk.json"
gdf = gpd.read_file(geojson_file)

DK1 = gdf[gdf['name'].str.contains('Nordjylland|Midtjylland|Syddanmark', case=False)].dissolve(by='name', aggfunc='sum')
DK1 = DK1.geometry.unary_union
DK1 = gpd.GeoDataFrame(geometry=[DK1])
DK1['Electricity area'] = 'DK1'

DK2 = gdf[gdf['name'].str.contains('Hovedstaden|Sjaælland', case=False)].dissolve(by='name', aggfunc='sum')
DK2 = DK2.geometry.unary_union
DK2 = gpd.GeoDataFrame(geometry=[DK2])
DK2['Electricity area'] = 'DK2'

DKdf = pd.concat([DK1, DK2], ignore_index=True)
DK = gpd.GeoDataFrame(DKdf, geometry='geometry')

color0 = ["#3468C0", "#5DEBD7"]
fig0 = px.choropleth(DK, 
                    geojson=DK.geometry, 
                    locations= DK.index, 
                    projection="mercator",
                    color='Electricity area',
                    color_discrete_sequence =color0,
                    hover_name=['DK1', 'DK2'],
                    hover_data= None
                    )

hover_templ = '%{hovertext}'
fig0.update_traces(hovertemplate=hover_templ)

fig0.update_geos(fitbounds="locations", visible=False)

fig0.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0}, 
       legend=dict(
           x=0.78,
           y=0.9, 
           traceorder='normal', 
           font = dict(size = 28, family="Raleway"),
           title="Electricity area's"
           ),
       dragmode=False
       
)






def generate_graph_and_table_layout(graph_fig, df,table_id, ):
    graph = dcc.Graph(
        id='graph',
        figure=graph_fig,
        className='regression2-graph'
    )
    
    table = dash_table.DataTable(
        id=table_id,
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'backgroundColor': 'honeydew', 'width': '80%'},  
        style_cell={'color': '#211C6A', 'fontFamily': 'Raleway', 'minWidth': '100px', 'maxWidth': '200px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'textAlign': 'left'},
        style_header={'backgroundColor': 'honeydew', 'fontWeight': 'bold', 'minWidth': '100px', 'maxWidth': '200px', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'textAlign': 'center', 'color': 'hotpink'},
        style_cell_conditional=[
            {'if': {'column_id': 'Error Type'}, 'width': '50%'},  
            {'if': {'column_id': 'Values'}, 'width': '30%', 'textAlign': 'center'}   
        ]
    )    
    return html.Div([graph, table], className='graph-and-table-container')






#PLOTLY DASH---------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['costumstyle.css'])
server = app.server

app.layout = html.Div(
    children=[
        html.H1(
            children='Denmark next day prediction for the CO2 emissions for electricity',
            style={'textAlign': 'center', 'color': 'white', 'fontSize': '80px', 'fontFamily': 'Raleway'}
        ),
        html.Div(
            dcc.Graph(
                id='dk-map',
                figure=fig0,
                config={'displaylogo': False, 'displayModeBar': False},
                style={'width': '100%', 'height': '880px', 'margin': '0 auto'}
            ),
            style={'border': '0px solid black', 'border-radius': '50px', 'overflow': 'hidden', 'margin': '40px'}
        ),
        html.Div(
            id='areas',
            children=[
                html.Div(
                    id='area-info',
                    children='Content for level 1 info',
                    style={'textAlign': 'center', 'color': 'white', 'fontFamily': 'Raleway', 'fontSize': '40px',
                           'paddingLeft': '40px', 'width': '60%', 'display': 'inline-block'}
                ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H3('Electricity generation in Denmark 2022', style={'textAlign': 'center', 'color': '#211C6A', 'fontFamily': 'Raleway', 'fontSize': '48px'}),
                        html.Div(
                            dcc.Graph(
                                id='electricity-pie-chart',
                                figure=fig2,
                                config={'displaylogo': False, 'displayModeBar': False},
                            ),
                        ),
                        html.Div(
                            dcc.Graph(
                                id='electricity-pie-chart2',
                                figure=fig3,
                                config={'displaylogo': False, 'displayModeBar': False},
                            ),
                        ),
                    ],
                    style={'border': '0px solid black', 'border-radius': '20px', 'overflow': 'hidden', 'margin': '0px','backgroundColor':'#F0FFF0'}
                ),
            ],
            style={'color': '#211C6A', 'width': '37%', 'display': 'inline-block', 'verticalAlign': 'top'}
        )],
        )
    ],
    style={'backgroundColor': '#211C6A'}
)

suppress_callbacks = False

@app.callback(
    Output('area-info', 'children'),
    [Input('dk-map', 'clickData')]
)
def update_area_info(clickData):
    if not suppress_callbacks:
        if clickData is not None:
            area_name = clickData['points'][0]['location']
            if area_name == 0:
                return html.Div([
                    html.H3('DK 1', style={'textAlign': 'center', 'color': '#FFFFFF', 'fontFamily': 'Raleway', 'fontSize': '48px'}),
                    html.Div([ 
                        dcc.Tabs(
                            id='tabs-with-classes',
                            value='tab-2', 
                            parent_className='custom-tabs', 
                            className='custom-tabs-container',
                            children=[ 
                                dcc.Tab(
                                    label='Linear Regression',
                                    value='tab-1',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK1LR',
                                            className='regression-graph', 
                                            figure=figDK1LR
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Support Vector',
                                    value='tab-2',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK1SV',
                                            className='regression-graph', 
                                            figure=figDK1SVR
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Decision Tree',
                                    value='tab-3',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK1DT',
                                            className='regression-graph', 
                                            figure=figDK1DT
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Random Forest',
                                    value='tab-4',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK1RF',
                                            className='regression-graph', 
                                            figure=figDK1RF
                                            )
                                    ]
                                ),             
                            ]),
                        dcc.Checklist(id='table-switch',
                                      options=[{'label': 'Show error data & scatter plot', 'value': 'show'}], 
                                      value=[],
                                      className='container',
                                     ),
                        html.Div(id='tabs-content-classes'),

                    ])
                ])
            
            elif area_name == 1:
                return html.Div([
                    html.H3('DK 2', style={'textAlign': 'center', 'color': '#FFFFFF', 'fontFamily': 'Raleway', 'fontSize': '48px'}),
                    html.Div([
                        dcc.Tabs(
                            id='tabs-with-classes',
                            value='tab-2', 
                            parent_className='custom-tabs', 
                            className='custom-tabs-container',
                            children=[ 
                                dcc.Tab(
                                    label='Linear Regression',
                                    value='tab2-1',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK2LR',
                                            className='regression-graph', 
                                            figure=figDK2LR
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Support Vector',
                                    value='tab2-2',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK2SV',
                                            className='regression-graph', 
                                            figure=figDK2SVR
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Decision Tree',
                                    value='tab2-3',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK2DT',
                                            className='regression-graph', 
                                            figure=figDK2DT
                                            )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Random Forest',
                                    value='tab2-4',
                                    className='custom-tab',
                                    selected_className='custom-tab--selected',
                                    children=[ 
                                        dcc.Graph( 
                                            id='DK2RF',
                                            className='regression-graph', 
                                            figure=figDK2RF
                                            )
                                    ]
                                ),             
                            ]),
                        dcc.Checklist(id='table-switch',
                                      options=[{'label': 'Show error data & scatter plot', 'value': 'show'}], 
                                      value=[],
                                      className='container',
                                     ),
                        html.Div(id='tabs-content-classes'),

                    ])
                ])
            
            else:
                return html.Div([])
        else:
            default_message = "CLICK ON THE MAP ON ONE OF THE AREA'S"
            return html.Div(
                id='clickme',
                children=[
                    html.Div(
                        children=[
                            html.H3(style={'textAlign': 'center', 'color': '#211C6A', 'fontFamily': 'Raleway',
                                           'fontSize': '10px'}),
                            html.Blockquote(default_message, className='animated-text1')
                        ],
                        style={'border': '0px solid black', 'border-radius': '20px', 'overflow': 'hidden',
                               'margin': '0px', 'background': '#FFFFF00'}
                    ),
                ],
                style={'textAlign': 'center', 'color': '#211C6A', 'width': '60%', 'display': 'inline-block',
                       'verticalAlign': 'top'}
            )

    else:
        return html.Div([])  

suppress_callbacks = False
@app.callback(
    Output('tabs-content-classes', 'children'),
    [Input('tabs-with-classes', 'value'),
     Input('table-switch', 'value'),
     Input('dk-map', 'clickData')]
)
def render_content(tab, switch_value, clickData):
    switch_on = 'show' in switch_value
    if clickData is not None:
        area_name = clickData['points'][0]['location']
        if area_name == 0:
            if tab == 'tab-1' and switch_on:
                df = ER_DK1[0] 
                return generate_graph_and_table_layout(figDK1LR_s, df, 'ERDK1_LR')
            elif tab == 'tab-2' and switch_on:
                df = ER_DK1[1]               
                return generate_graph_and_table_layout(figDK1SVR_s,df, 'ERDK1_SV')
            elif tab == 'tab-3' and switch_on:
                df = ER_DK1[2]  
                return generate_graph_and_table_layout(figDK1RF_s,df, 'ERDK1_RF')
            elif tab == 'tab-4' and switch_on:
                df = ER_DK1[3]  
                return generate_graph_and_table_layout(figDK1DT_s,df, 'ERDK1_DT')
            else:
                return html.Div([]) 

        elif area_name == 1:
            if tab == 'tab2-1' and switch_on:
                df = ER_DK2[0]  
                return generate_graph_and_table_layout(figDK2LR_s,df, 'ERDK2_LR')
            elif tab == 'tab2-2' and switch_on:
                df = ER_DK2[1]  
                return generate_graph_and_table_layout(figDK2SVR_s,df, 'ERDK2_SV')
            elif tab == 'tab2-3' and switch_on:
                df = ER_DK2[2]  
                return generate_graph_and_table_layout(figDK2RF_s,df, 'ERDK2_RF')
            elif tab == 'tab2-4' and switch_on:
                df = ER_DK2[3]  
                return generate_graph_and_table_layout(figDK2DT_s,df, 'ERDK2_DT')
            else:
                return html.Div([]) 
        else:
            return html.Div([])
    else:
        return html.Div([])

if __name__ == '__main__':
    app.run_server()
