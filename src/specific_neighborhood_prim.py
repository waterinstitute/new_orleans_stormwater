import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import prim
#-------------------------------- create a dataset with the neighborhood vulnerability in different scenarios 

hour24_neighbor =  pd.read_csv ("data/NO_storm_damage_by_neighborhood_long_24_hours_updated.csv")
neighbor_structure_value = pd.read_csv ("data/No_value_by_neighborhood_long.csv")
neighborhoods = gpd.read_file("data/neighborhood_shp/Neighborhood_Statistical_Areas.shp")

print(hour24_neighbor.head())
for column_headers in hour24_neighbor.columns: 
    print(column_headers)

print(hour24_neighbor['condition'].unique())

#hour24_neighbor = hour24_neighbor.loc[(hour24_neighbor['recurrence'] == 'EAD') & (hour24_neighbor['threshold'] == '>12in') & (hour24_neighbor['NB'] == 'SIXTH WARD - TREME - LAFITTE')]
hour24_neighbor = hour24_neighbor.loc[(hour24_neighbor['recurrence'] == 'EAD') & (hour24_neighbor['threshold'] == '>12in') & (hour24_neighbor['NB'] == 'MILNEBURG')]
#hour24_neighbor = hour24_neighbor.loc[(hour24_neighbor['recurrence'] == 'EAD') & (hour24_neighbor['threshold'] == '>12in') & (hour24_neighbor['NB'] == 'BROADMOOR')]
print(hour24_neighbor.head())

hour24_neighbor=hour24_neighbor.drop(["recurrence", "threshold"], axis = 1)
hour24_neighbor = hour24_neighbor.groupby(["condition"]).damage.sum().reset_index()



#neighbor_structure_value = neighbor_structure_value.loc[(neighbor_structure_value['NB'] == 'SIXTH WARD - TREME - LAFITTE')]
neighbor_structure_value = neighbor_structure_value.loc[(neighbor_structure_value['NB'] == 'MILNEBURG')]
#neighbor_structure_value = neighbor_structure_value.loc[(neighbor_structure_value['NB'] == 'BROADMOOR'


neighbor_structure_value.to_csv('outputs/structur_value_boraodmoor_before.csv')
neighbor_structure_value = neighbor_structure_value.groupby(["NB"]).value.sum().reset_index()

# print("structure value is 0:", (neighbor_structure_value['value'] == 0).sum())
# print("rows wit structure value:-----------------", neighbor_structure_value.loc[neighbor_structure_value['value'] == 0])
# neighbor_structure_value = neighbor_structure_value[neighbor_structure_value.value != 0]

neighbor_structure_value_sum=neighbor_structure_value.iloc[0]['value']
print ('neighbor_structure_value_sum-----',neighbor_structure_value_sum)
neighbor_structure_value.to_csv('outputs/structur_value_boraodmoor.csv')


results = hour24_neighbor


results['damage_percent'] = (results ['damage']/neighbor_structure_value_sum)*100

results['vulnerable'] = np.where(results['damage_percent']>2, True, False)

print(results['vulnerable'].value_counts())
print('results describe------',results.describe())
print('results length------',len(results))

print(results.isna().sum())


#--------------------------------------------------- Prepare data for PRIM
scenario_df= pd.read_csv ("data/NO Storm Water Design Updated.csv")
for column_headers in scenario_df.columns: 
     print(column_headers)
print('len---------',len(scenario_df))

#---------------------------------- ------------uncomment this for all-on scenarios
# scenario_df = scenario_df[scenario_df["full.description"] == "All On"]
# scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness']]


scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]


print(scenario_df.head())
print(len(scenario_df))
#scenario_df['Scenario'] ='scenario' + scenario_df['Scenario'].astype(str)
for column_headers in scenario_df.columns: 
     print(column_headers)

scenario_df['condition'] =scenario_df['Scenario']

df= results
df=df[['condition','vulnerable']]
print(df.head())

prim_data= df.merge(scenario_df, left_on="condition", right_on="condition")
print(prim_data.head())

for column_headers in prim_data.columns: 
     print(column_headers)

print(len(prim_data))

prim_data.drop (['condition','condition'], axis=1, inplace=True)
prim_data.drop (['Scenario'], axis=1, inplace=True)
prim_data['SLR'] ='SLR' + prim_data['SLR'].astype(str)
#prim_data['Pump.State'] ='Pump_State' + prim_data['Pump.State'].astype(str)
prim_data.to_csv('data/prim_data.csv')

#-----------------------------------------------------------------------------------------PRIM analysis 



#x = prim_data[['rainfall', 'SLR', 'roughness']]
#x = prim_data[['rainfall', 'SLR', 'roughness', 'Pump.State']]
x = prim_data[['rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]
vulnerability_criterion = prim_data['vulnerable']



prim_alg = prim.Prim(x, vulnerability_criterion, threshold=0, threshold_type=">")
box = prim_alg.find_box()
box_df=box.limits
# print(df.head())
box_df.to_csv('outputs/prim_box_BROADMOOR_allScenario_df.csv')
box.show_tradeoff()
box.show_details()
print(box)
plt.show()