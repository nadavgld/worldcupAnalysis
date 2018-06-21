import util
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

raw_data = util.get_csv()


home_target = raw_data["Home_Score"]
away_target = raw_data["Away_Score"]
gamesPlayedIndex = home_target.count()

def teamSign(home,away):
    if (home > away):
        return 1
    elif (away > home):
        return 3
    else:
        return 2

def explain(score):
    if (score==1):
        return "home wins"
    elif (score==2):
        return "draw"
    elif (score == 3):
        return "away wins"
    return "nadav"


raw_data['WDW_tri'] = raw_data.apply(lambda row: teamSign(row['WDW_ Home'], row['WDW_Away']), axis=1)
raw_data['PDRZ_tri'] = raw_data.apply(lambda row: teamSign(row['PRDZ_Home'], row['PRDZ_Away']), axis=1)
raw_data['FORE_tri'] = raw_data.apply(lambda row: teamSign(row['FORE_Home'], row['FORE_Away']), axis=1)
raw_data['VITI_tri'] = raw_data.apply(lambda row: teamSign(row['VITI_Home'], row['VITI_Away']), axis=1)
raw_data['FREESUPERTIPS_tri'] = raw_data.apply(lambda row: teamSign(row['FREESUPERTIPS_Home'], row['FREESUPERTIPS_Away']), axis=1)
raw_data['SCOREPREDICTOR_tri'] = raw_data.apply(lambda row: teamSign(row['SCOREPREDICTOR_Home'], row['SCOREPREDICTOR_Away']), axis=1)
raw_data['target_tri'] = raw_data.apply(lambda row: teamSign(row['Home_Score'], row['Away_Score']), axis=1)

model_var = raw_data[['WDW_tri','PDRZ_tri','FORE_tri','VITI_tri', 'FREESUPERTIPS_tri','SCOREPREDICTOR_tri']]
#print(raw_data[['WDW_ Home', 'WDW_Away', 'WDW_tri']])

train_var = model_var.iloc[:gamesPlayedIndex, ]
train_target = raw_data['target_tri'].iloc[:gamesPlayedIndex, ]
pred_var = model_var.iloc[gamesPlayedIndex:gamesPlayedIndex + 1, ]

#random forest model
rf_class = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=2, min_samples_split=5).fit(train_var, train_target)
rf_result = rf_class.predict(pred_var)
rf_prob = rf_class.predict_proba(pred_var)
print "rf class: ", explain(rf_result)
print " home prob.: ", round(rf_prob[0][0],2), " away prob.: ", round(rf_prob[0][2],2), " draw prob.: ", round(rf_prob[0][1],2)

#KNN
knn_class = KNeighborsClassifier(n_neighbors=3, weights='distance').fit(train_var, train_target)
knn_results = knn_class.predict(pred_var)
knn_prob = knn_class.predict_proba(pred_var)
print "knn class: ", explain(knn_results)
print " home prob.: ", round(knn_prob[0][0],2), " away prob.: ", round(knn_prob[0][2],2), " draw prob.: ", round(knn_prob[0][1],2)

svc_class = LinearSVC().fit(train_var, train_target)
svc_results = svc_class.predict(pred_var)
#svc_prob = svc_class.predict_proba(pred_var)
print "svc class: ", explain(svc_results)
#print " home prob.: ", round(svc_prob[0][0],2), " away prob.: ", round(svc_prob[0][2],2), " draw prob.: ", round(svc_prob[0][1],2)