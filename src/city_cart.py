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


#--------------------------------------------------- Prepare data for CART
scenario_df= pd.read_csv ("data/NO Storm Water Design Updated.csv")
for column_headers in scenario_df.columns: 
     print(column_headers)
print('len---------',len(scenario_df))
#---------------------------------- ------------uncomment this for all-on scenarios 
scenario_df = scenario_df[scenario_df["full.description"] == "All On"]
scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness']]
print('len---------',len(scenario_df))

#scenario_df=scenario_df[['Scenario','rainfall','SLR','roughness','DPS.1','DPS.Pritchard','DPS.Orleander','DPS.6','X17th.Street.Canal.Closure.Pump','DPS.I10','DPS.12','DPS.2','DPS.7','Orleans.Ave.Canal.Closure.Pump','DPS.3','DPS.4','Lond.Ave..Canal.Closure.Pump','DPS.17','DPS.19','DPS.18','DPS.15','DPS.20','DPS.Elaine','DPS.Grant','DPS.Dwyer','DPS.16','DPS.10','DPS.14','DPS.5','DPS.11','DPS.13']]
print(scenario_df.head())
print(len(scenario_df))
#scenario_df['Scenario'] ='scenario' + scenario_df['Scenario'].astype(str)
for column_headers in scenario_df.columns: 
     print(column_headers)

scenario_df['condition'] =scenario_df['Scenario']

df= results
df=df[['condition','vulnerable', 'NB']]
print(df.head())

city_cart_data= df.merge(scenario_df, left_on="condition", right_on="condition")
print(city_cart_data.head())

for column_headers in city_cart_data.columns: 
     print(column_headers)

print(len(city_cart_data))

city_cart_data.drop (['condition','condition'], axis=1, inplace=True)

rainfall=pd.get_dummies(city_cart_data['rainfall'], drop_first=False)
SLR=pd.get_dummies(city_cart_data['SLR'], drop_first=False)
roughness=pd.get_dummies(city_cart_data['roughness'], drop_first=False)
NB=pd.get_dummies(city_cart_data['NB'], drop_first=False)



city_cart_data.drop (['rainfall','SLR','roughness', 'NB'], axis=1, inplace=True)
city_cart_data=pd.concat([city_cart_data,rainfall,SLR,roughness, NB], axis=1)



labels = pd.DataFrame(city_cart_data['vulnerable'])
city_cart_data=city_cart_data.drop(['vulnerable'], axis=1)
print(city_cart_data.head())
city_cart_data=city_cart_data.apply(pd.to_numeric)
labels=labels.apply(pd.to_numeric)


city_cart_data.drop (['Scenario'], axis=1, inplace=True)


#-------------------------------------------------------------------------------------- CART


def fitting(X, y, criterion, splitter, mdepth, clweight, minleaf):

    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion, 
                                            splitter=splitter, 
                                            max_depth=mdepth,
                                            class_weight=clweight,
                                            min_samples_leaf=minleaf, 
                                            random_state=0, 
                                    )
    clf = model.fit(X_train, y_train)

    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

        # Tree summary and model evaluation metrics
    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of leaves: ', clf.tree_.n_leaves)
    #print('No. of features: ', clf.n_features_in_)
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

        # Use graphviz to plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=X.columns, 
                                    class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                    filled=True, 
                                    rounded=True, 
                                    #rotate=True,
                                ) 
    graph = graphviz.Source(dot_data)

    return X_train, X_test, y_train, y_test, clf, graph



X_train, X_test, y_train, y_test, clf, graph = fitting(city_cart_data, labels,  'gini', 'best', 
                                                       mdepth=5, 
                                                       clweight=None,
                                                       minleaf=50)

# Plot the tree graph
graph

graph.render('outputs/Decision_Tree_all_vars_gini_city_all_scenarios_depth3')



plt.show()
