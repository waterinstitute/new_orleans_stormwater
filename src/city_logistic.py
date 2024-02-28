import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree # for decision tree models

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs
#-------------------------------- create a dataset with neighborhood vulnerability in different scenarios 

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


#--------------------------------------------------- Prepare data for Logistic regression 
scenario_df= pd.read_csv ("data/NO Storm Water Design Updated.csv")
for column_headers in scenario_df.columns: 
     print(column_headers)
print('len---------',len(scenario_df))
#---------------------------------- ------------uncomment this for all-on scenarios 
# scenario_df = scenario_df[scenario_df["full.description"] == "All On"]
# scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness']]
# # print('len---------',len(scenario_df))

scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]
print(scenario_df.head())
print(len(scenario_df))
#scenario_df['Scenario'] ='scenario' + scenario_df['Scenario'].astype(str)
for column_headers in scenario_df.columns: 
     print(column_headers)

scenario_df['condition'] =scenario_df['Scenario']

df= results
df=df[['condition','vulnerable', 'NB']]
print(df.head())

city_logistic_data= df.merge(scenario_df, left_on="condition", right_on="condition")
print(city_logistic_data.head())

for column_headers in city_logistic_data.columns: 
     print(column_headers)

print(len(city_logistic_data))

city_logistic_data.drop (['condition','condition'], axis=1, inplace=True)

rainfall=pd.get_dummies(city_logistic_data['rainfall'], drop_first=False)
SLR=pd.get_dummies(city_logistic_data['SLR'], drop_first=False)
roughness=pd.get_dummies(city_logistic_data['roughness'], drop_first=False)
NB=pd.get_dummies(city_logistic_data['NB'], drop_first=False)



city_logistic_data.drop (['rainfall','SLR','roughness', 'NB'], axis=1, inplace=True)
city_logistic_data=pd.concat([city_logistic_data,rainfall,SLR,roughness, NB], axis=1)



labels = pd.DataFrame(city_logistic_data['vulnerable'])
city_logistic_data=city_logistic_data.drop(['vulnerable'], axis=1)
print(city_logistic_data.head())
city_logistic_data=city_logistic_data.apply(pd.to_numeric)
labels=labels.apply(pd.to_numeric)


city_logistic_data.drop (['Scenario'], axis=1, inplace=True)


#---------------------------------------------------------------Logistic regression 



X_train, X_test, y_train, y_test = train_test_split(city_logistic_data, labels, test_size=0.3, random_state=0)

logmodel = LogisticRegression(class_weight='balanced')
logmodel.fit(X_train, y_train)
predictions_log = logmodel.predict (X_test)

print(classification_report(y_test, predictions_log))

importance = logmodel.coef_.flatten()
#importance = importance.astype(str)
print(importance)

features=city_logistic_data.columns.astype(str)

importance_df = pd.DataFrame (importance, columns = ['importance'], index=features)
print (importance_df)
importance_df['importance_abs']= importance_df['importance'].abs()
importance_df=importance_df.sort_values('importance_abs',  ascending=[False])
importance_df.to_csv('outputs/feature_importance_all_on_city.csv')

print(city_logistic_data.columns)
plt.rcParams["figure.figsize"]= (10,10)
plt.rcParams.update({'font.size': 6})
plt.barh(city_logistic_data.columns.astype(str), importance, color = 'g')
plt.title("summary of feature importance for neighborhood vulnerability")
plt.xlabel ("score")

# fig, ax = plt.subplots()
# scatter=ax.scatter(df.columns, importance, c='g')


plt.show()
