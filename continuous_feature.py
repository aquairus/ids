import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation


from sklearn.linear_model import LogisticRegressionCV,LogisticRegression


df=pd.read_csv("data/kddcup_data_10_percent.txt")
select_feature=[1,5,6,8,9,10,11,13,16,17,18,19,20,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,]


def normal_or_not(text):
    if text=="normal.":
        return 1
    else:
        return 0

washed_df=pd.DataFrame()

washed_df[0]=df['normal.'].map(normal_or_not)
for i in select_feature:
    washed_df[i]=df.iloc[:,i-1]



train_data = washed_df.values



logr=LogisticRegression(n_jobs=-1)
log_cv=LogisticRegressionCV()


logr=log_cv.fit(train_data[0::,1::],train_data[0::,0] )


print logr.score(train_data[0::,1::],y_true)
