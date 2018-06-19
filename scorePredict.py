import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import util

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
rf_home_scoreModel = RandomForestRegressor(n_estimators=20, criterion='mae').fit(home_var, home_target)
rf_away_scoreModel = RandomForestRegressor(n_estimators=20, criterion='mae').fit(away_var, away_target)
rf_home_pred = rf_home_scoreModel.predict(home_pred)
rf_away_pred = rf_away_scoreModel.predict(away_pred)
print("random forest: ", rf_home_pred[0], rf_away_pred[0])

#LinearRegression
lr_home_scoreModel = LinearRegression().fit(home_var, home_target)
lr_away_scoreModel = LinearRegression().fit(away_var, away_target)
lr_home_pred = lr_home_scoreModel.predict(home_pred)
lr_away_pred = lr_away_scoreModel.predict(away_pred)
print("linear regression: ", lr_home_pred[0], lr_away_pred[0])