import pandas as pd
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn import cross_validation, datasets, metrics, tree, ensemble
%pylab inline


data=pd.read_csv('Titanic/train.csv')
test=pd.read_csv('Titanic/test.csv')

data.shape
data.head()

y=data['Survived']
x=data.ix[:,2:]


x.head()



x_tr,x_ts,y_tr,y_ts=cross_validation.train_test_split(x,y,test_size = 0.3)






#----- model ------

#DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(random_state = 1, max_depth = 15)
clf.fit(x_tr, y_tr)

predictions = clf.predict(x_ts)
metrics.accuracy_score(y_ts, predictions)

cross_validation.cross_val_score(clf, x, y, scoring = 'accuracy', cv = 3).mean()