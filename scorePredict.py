from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import util
import winnerPredict
from keras.models import Sequential
from keras.layers import Dense
import os

os.system("cls")


def run(gamesToPredict):
    raw_data = util.get_csv()

    # gamesToPredict = 4
    # target data
    home_target = raw_data["Home_Score"]
    away_target = raw_data["Away_Score"]

    # data to analyse
    home_data = ['WDW_ Home', 'PRDZ_Home', 'FORE_Home', 'VITI_Home', 'FREESUPERTIPS_Home', 'SCOREPREDICTOR_Home']
    away_data = ["WDW_Away", "PRDZ_Away", "FORE_Away", "VITI_Away", "FREESUPERTIPS_Away", "SCOREPREDICTOR_Away"]
    home_var = raw_data[home_data]
    away_var = raw_data[away_data]

    # only played games to build the model
    gamesPlayedIndex = home_target.count()

    home_target = home_target.iloc[:gamesPlayedIndex, ]
    away_target = away_target.iloc[:gamesPlayedIndex, ]
    home_pred = home_var.iloc[gamesPlayedIndex:gamesPlayedIndex + gamesToPredict, ]
    away_pred = away_var.iloc[gamesPlayedIndex:gamesPlayedIndex + gamesToPredict, ]
    home_var = home_var.iloc[:gamesPlayedIndex, ]
    away_var = away_var.iloc[:gamesPlayedIndex, ]

    currentGame = raw_data[gamesPlayedIndex:gamesPlayedIndex + gamesToPredict]

    for i in range(0, gamesToPredict):
        print "\n    " + currentGame['Home'].values[i] + " vs " + currentGame['Away'].values[i]

        # models

        # random forest
        rf_home_scoreModel = RandomForestRegressor(n_estimators=100, criterion='mae').fit(home_var, home_target)
        rf_away_scoreModel = RandomForestRegressor(n_estimators=100, criterion='mae').fit(away_var, away_target)
        rf_home_pred = rf_home_scoreModel.predict(home_pred)
        rf_away_pred = rf_away_scoreModel.predict(away_pred)
        print "\nRandom-Forest: \n\t", int(round(rf_home_pred[i])), "-", int(
            round(rf_away_pred[i])), "\n w/o round: ", round(
            rf_home_pred[i], 2), "-", round(rf_away_pred[i], 2)

        # LinearRegression
        lr_home_scoreModel = LinearRegression().fit(home_var, home_target)
        lr_away_scoreModel = LinearRegression().fit(away_var, away_target)
        lr_home_pred = lr_home_scoreModel.predict(home_pred)
        lr_away_pred = lr_away_scoreModel.predict(away_pred)
        print "\nLinear-Regression:\n\t", int(round(lr_home_pred[i])), "-", int(
            round(lr_away_pred[i])), "\n w/o round: ", round(lr_home_pred[i], 2), "-", round(lr_away_pred[i], 2)

        # nn
        # def baseline_model():
        #     nn = Sequential()
        #     nn.add(Dense(10, input_dim=6, activation='relu'))
        #     nn.add(Dense(1))
        #     nn.compile(loss='mean_squared_error', optimizer='adam')
        #     return nn

        # nn_est = KerasRegressor(build_fn=baseline_model, epochs = 100, batch_size=3)
        # nn_est.fit(home_var, home_target)
        # nn_est.predict(home_pred)
        # print(nn_est.predict(home_pred))

        print("\nnn-model:")
        model = Sequential()
        model.add(Dense(4, input_dim=6, activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        model.fit(home_var, home_target, epochs=300, verbose=0)
        nn_home = model.predict(home_pred)
        model.fit(away_var, away_target, epochs=300, verbose=0)
        nn_away = model.predict(away_pred)

        f_home = int(round(float(str(nn_home[i][0]))))
        f_away = int(round(float(str(nn_away[i][0]))))
        print "\t" + str(f_home) + " - " + str(f_away), "\n w/o round: ", str(round(nn_home[i][0], 2)), "-", str(
            round(nn_away[i][0], 2))

        winnerPredict.prepare_data(raw_data)

        print "\n----------------"


gamesToPredict = raw_input("Amount of future games to predict? \t")
run(int(gamesToPredict))

rerun = raw_input("Re-run?(y/n)")
while rerun.lower() == "y":
    gamesToPredict = raw_input("Amount of future games to predict? \t")
    run(int(gamesToPredict))
    rerun = raw_input("Re-run?(y/n)")

print "byebye:)"
