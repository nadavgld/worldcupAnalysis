import pandas as pd

raw_data = pd.read_csv("WorldCup 2018.csv")
#print(raw_data)

home_target = raw_data["Home_Score"]
away_target = raw_data["Away_Score"]

home_data = ['WDW_ Home', 'PRDZ_Home', 'FORE_Home', 'VITI_Home', 'FREESUPERTIPS_Home', 'SCOREPREDICTOR_Home']
away_data = ["WDW_Away", "PRDZ_Away", "FORE_Away", "VITI_Away", "FREESUPERTIPS_Away", "SCOREPREDICTOR_Away"]
home_var = raw_data[home_data]
away_var = raw_data[away_data]

