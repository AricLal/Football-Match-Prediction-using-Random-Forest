import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
matches= pd.read_csv("matches.csv",index_col=0)
# print(matches.head())
# print(matches.shape)
# print(matches["team"].value_counts())
# print(matches[matches["team"]=="Liverpool"])
# print(matches["round"].value_counts())

matches["date"]=pd.to_datetime(matches["date"])
# print(matches.dtypes)
#
# print(matches)

class MissingDict(dict):
    __missing__=lambda self,key:key

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group.set_index("date")[cols].rolling(3,closed="left").mean()  # Use "date" as index and specify "3D" for 3-day rolling window
    group[new_cols] = rolling_stats.values  # Assign values back to the original DataFrame
    group = group.dropna(subset=new_cols)
    return group

def make_predictions(data,predictors):
    train=data[data["date"]<"2022-01-01"]
    test = data[data["date"] > "2022-01-01"]
    rf.fit(train[predictors],train["target"])
    preds=rf.predict(test[predictors])
    combined=pd.DataFrame(dict(actual=test["target"],predicted=preds),index=test.index)
    precision=precision_score(test["target"],preds)
    return combined, precision


matches["venue_code"]=matches["venue"].astype("category").cat.codes
matches["opp_code"]=matches["opponent"].astype("category").cat.codes
matches["hour"]=matches["time"].str.replace(":.+","",regex=True).astype("int")
matches["day_code"]=matches["date"].dt.dayofweek
matches["target"]=(matches["result"]=="W").astype("int")

rf=RandomForestClassifier(n_estimators=50,min_samples_split=10, random_state=1)

train=matches[matches["date"]<"2022-01-01"]
test=matches[matches["date"]>"2022-01-01"]
predictors=["venue_code","opp_code","hour","day_code"]
rf.fit(train[predictors],train["target"])

preds=rf.predict(test[predictors])
acc=accuracy_score(test["target"],preds)
print(acc)
combined=pd.DataFrame(dict(actual=test["target"],predicted=preds))
print(pd.crosstab(index=combined["actual"],columns=combined["predicted"]))

print(precision_score(test["target"],preds))

grouped_matches=matches.groupby("team")
group=grouped_matches.get_group("Manchester City")
print(group)

cols=["gf","ga","sh","sot","dist","fk","pk","pkatt"]
new_cols=[f"{c}_rolling" for c in cols]
print(new_cols)

print(rolling_averages(group,cols,new_cols))

matches_rolling=matches.groupby("team").apply(lambda x: rolling_averages(x,cols,new_cols))
matches_rolling=matches_rolling.droplevel("team")
print(matches_rolling)

matches_rolling.index=range(matches_rolling.shape[0])
print(matches_rolling)

combined,precision=make_predictions(matches_rolling,predictors+new_cols)
print(precision)
print(combined)
combined=combined.merge(matches_rolling[["date","team","opponent","result"]],left_index=True,right_index=True)
print(combined)

map_values={
    "Brighton and Hove Albion":"Brighton",
    "Manchester United":"Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur":"Tottenham",
    "West Ham United":"West Ham",
    "Wolverhampton Wanderers":"Wolves"
}
mapping=MissingDict(**map_values)
print(mapping["West Ham United"])
combined["new_team"]=combined["team"].map(mapping)
print(combined)

merged=combined.merge(combined,left_on=["date","new_team"],right_on=["date","opponent"])
print(merged)

print(merged[(merged["predicted_x"]==1) & (merged["predicted_y"]==0)]["actual_x"].value_counts())