import pandas as pd
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn import cross_validation, datasets, metrics, tree, ensemble
%pylab inline


data=pd.read_csv('Digit Recognizer/train.csv')
test=pd.read_csv('Digit Recognizer/test.csv')

data.head()
data.shape

#Form data
y=data['label']
x=data.ix[:,1:]

x_tr,x_ts,y_tr,y_ts=cross_validation.train_test_split(x,y,test_size = 0.3)


#----- model ------

#DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(random_state = 1, max_depth = 15)
clf.fit(x_tr, y_tr)

predictions = clf.predict(x_ts)
metrics.accuracy_score(y_ts, predictions)

cross_validation.cross_val_score(clf, x, y, scoring = 'accuracy', cv = 3).mean()

#RandomForestClassifier
clf = ensemble.RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = 1)

clf.fit(x_tr, y_tr)

predictions = clf.predict(x_ts)
metrics.accuracy_score(y_ts, predictions)
cross_validation.cross_val_score(clf, x, y, scoring = 'accuracy', cv = 3).mean()


#XGBoots
import xgboost as xgb
clf = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=20, min_child_weight=4)
clf = xgb.XGBClassifier()
clf.fit(x_tr, y_tr)

predictions = clf.predict(x_ts)
metrics.accuracy_score(y_ts, predictions)

cross_validation.cross_val_score(clf, x, y, scoring = 'accuracy', cv = 3).mean()



#test file
test=pd.read_csv('Digit Recognizer/test.csv')
test.shape

answer = clf.predict(test)


anw.head()
anw.index+=1
anw=pd.DataFrame({'Label':answer})
anw.index.rename('ImageId', inplace=True)

anw.to_csv('submission.csv', sep=',', header=True)