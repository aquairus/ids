import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("data/kddcup_data_10_percent.txt")

vectorize_map={}
select_feature=[2, 3 ,4 ,7  ,12 ,14 ,15 ,21 ,22]
for i in select_feature:
    select_key=df.columns[i-1]
    tcp_dict=dict(enumerate(np.unique(df[select_key])))
    vectorize_map[i]=dict((v,k) for k,v in tcp_dict.iteritems())


def normal_or_not(text):
    if text=="normal.":
        return 1
    else:
        return 0

washed_df=pd.DataFrame()

washed_df[0]=df['normal.'].map(normal_or_not)
for i in select_feature:
    key=df.columns[i-1]
    # print key
    washed_df[i]=df[key].map(vectorize_map[i])


print len(washed_df.loc[washed_df[0]==1])
print len (washed_df)
print len(washed_df.loc[washed_df[0]==1])/ float(len (washed_df))



from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10)
train_data = washed_df.values
forest = forest.fit(train_data[0::,1::],train_data[0::,0] )

from sklearn import cross_validation
forest_scores = cross_validation.cross_val_score(forest,train_data[0::,1::], train_data[0::,0],cv=5)
print forest_scores.mean()



from sklearn import metrics
from sklearn.metrics import precision_recall_curve
y_true = train_data[0::,0]
y_scores = forest.predict_proba(train_data[0::,1::])

precision, recall, thresholds = precision_recall_curve(y_true, y_scores[0::,1])


plt.plot( recall,precision)
print metrics.auc(recall,precision)


fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores[0::,1])
plt.plot( fpr, tpr)
print metrics.auc(fpr, tpr)
print forest.score(train_data[0::,:-1],y_true)
