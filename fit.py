#training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import cdsw

import csv as csv
from sklearn.ensemble import RandomForestClassifier
import pickle


test=pd.read_csv('preprocessed_test')
train=pd.read_csv('preprocessed_train')
target = train['Survived']
train_data = train.drop('Survived', axis=1)


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)


clf.fit(train_data, target)




# accuracy
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
accuracy=round(np.mean(score)*100, 2)
cdsw.track_metric("accuracy", accuracy)
print("accuracyr: %.2f"% accuracy)


# Output
filename = 'model.pkl'
pickle.dump(clf, open(filename, 'wb'))
cdsw.track_file(filename)


print(train_data)