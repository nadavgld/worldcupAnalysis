import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import util
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils

# raw_data = pd.read_csv("WorldCup 2018.csv")
raw_data = util.get_csv()

#target data
home_target = raw_data["Home_Score"]
away_target = raw_data["Away_Score"]

#data to analyse
home_data = ['WDW_ Home', 'PRDZ_Home', 'FORE_Home', 'VITI_Home', 'FREESUPERTIPS_Home', 'SCOREPREDICTOR_Home']
away_data = ["WDW_Away", "PRDZ_Away", "FORE_Away", "VITI_Away", "FREESUPERTIPS_Away", "SCOREPREDICTOR_Away"]
home_var = raw_data[home_data]
away_var = raw_data[away_data]

#only played games to build the model
gamesPlayedIndex = home_target.count()

home_target = home_target.iloc[:gamesPlayedIndex,]
away_target = away_target.iloc[:gamesPlayedIndex,]
home_pred = home_var.iloc[gamesPlayedIndex:gamesPlayedIndex+1,]
away_pred = away_var.iloc[gamesPlayedIndex:gamesPlayedIndex+1,]
home_var = home_var.iloc[:gamesPlayedIndex,]
away_var = away_var.iloc[:gamesPlayedIndex,]


#models

#random forest
rf_home_scoreModel = RandomForestRegressor(n_estimators=100, criterion='mae').fit(home_var, home_target)
rf_away_scoreModel = RandomForestRegressor(n_estimators=100, criterion='mae').fit(away_var, away_target)
rf_home_pred = rf_home_scoreModel.predict(home_pred)
rf_away_pred = rf_away_scoreModel.predict(away_pred)
print("random forest: ", rf_home_pred[0], rf_away_pred[0])

#LinearRegression
lr_home_scoreModel = LinearRegression().fit(home_var, home_target)
lr_away_scoreModel = LinearRegression().fit(away_var, away_target)
lr_home_pred = lr_home_scoreModel.predict(home_pred)
lr_away_pred = lr_away_scoreModel.predict(away_pred)
print("linear regression: ", lr_home_pred[0], lr_away_pred[0])

#nn
# def baseline_model():
#     nn = Sequential()
#     nn.add(Dense(10, input_dim=6, activation='relu'))
#     nn.add(Dense(1))
#     nn.compile(loss='mean_squared_error', optimizer='adam')
#     return nn

#nn_est = KerasRegressor(build_fn=baseline_model, epochs = 100, batch_size=3)
#nn_est.fit(home_var, home_target)
#nn_est.predict(home_pred)
#print(nn_est.predict(home_pred))

print("nn model")
model = Sequential()
model.add(Dense(4, input_dim=6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(home_var, home_target, epochs=300, verbose=0)
nn_home = model.predict(home_pred)
model.fit(away_var, away_target, epochs=300, verbose=0)
nn_away = model.predict(away_pred)
print(nn_home[0])
print(nn_away[0])