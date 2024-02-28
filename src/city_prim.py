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

print(neighbor_structure_value.head())
print(neighborhoods.head())
print(neighborhoods.head())
for column_headers in neighborhoods.columns: 
    print(column_headers)


print(hour24_neighbor.head())
for column_headers in hour24_neighbor.columns: 
    print(column_headers)

print(hour24_neighbor.describe())

print(hour24_neighbor['st_damcat'].value_counts())

hour24_neighbor = hour24_neighbor.loc[(hour24_neighbor['recurrence'] == 'EAD') & (hour24_neighbor['threshold'] == '>12in') ]
print(hour24_neighbor.head())

print(hour24_neighbor ['New Scenario ID'])
hour24_neighbor.to_csv('outputs/neighborhood.csv') 

hour24_neighbor=hour24_neighbor.drop(["recurrence", "threshold"], axis = 1)

# print(neighbor_structure_value.head())
# print(neighbor_structure_value['st_damcat'].value_counts())
# print(neighbor_structure_value.describe())
# print(neighbor_structure_value.isna().sum())
print(hour24_neighbor.isna().sum())

print("structure value is 0:", (neighbor_structure_value['value'] == 0).sum())


print("df:-----------------", neighbor_structure_value.loc[neighbor_structure_value['value'] == 0])

neighbor_structure_value = neighbor_structure_value[neighbor_structure_value.value != 0]


results = hour24_neighbor.merge(neighbor_structure_value, left_on=["NB", 'st_damcat'], right_on=["NB", 'st_damcat'])
results.to_csv('outputs/results.csv')

results=results.groupby(["NB", "condition"], as_index=False)["damage", "value"].apply(lambda x : x.astype(int).sum())
results.to_csv('outputs/results2.csv')

for column_headers in results.columns: 
    print(column_headers)

results['damage_percent'] = (results ['damage']/results ['value'])*100
results['vulnerable'] = np.where(results['damage_percent']>2, True, False)


print(results['vulnerable'].value_counts())
print(results.describe())

print(results.isna().sum())


#--------------------------------------------------- Prepare data for PRIM
scenario_df= pd.read_csv ("data/NO Storm Water Design Updated.csv")
for column_headers in scenario_df.columns: 
     print(column_headers)
print('len---------',len(scenario_df))

#---------------------------------- ------------uncomment this for all-on scenarios
scenario_df = scenario_df[scenario_df["full.description"] == "All On"]
scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness']]


#scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]


print(scenario_df.head())
print(len(scenario_df))
#scenario_df['Scenario'] ='scenario' + scenario_df['Scenario'].astype(str)
for column_headers in scenario_df.columns: 
     print(column_headers)

scenario_df['condition'] =scenario_df['Scenario']

#df= results
df=results[['condition','NB','vulnerable']]
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



x = prim_data[['rainfall', 'SLR', 'roughness', 'NB']]
#x = prim_data[['rainfall', 'SLR', 'roughness']]
#x = prim_data[['rainfall', 'SLR', 'roughness', 'Pump.State']]

#x = prim_data[['rainfall','SLR','roughness','NB','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]
#x = prim_data[['rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]
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