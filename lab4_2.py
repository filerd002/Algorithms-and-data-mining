import numpy as np
import os

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

global X_test
global y_test
global X_train
global y_train

def getDataset():
    global X_test
    global y_test
    global X_train
    global y_train

    X_test = np.loadtxt(os.path.normpath('in/Test/X_test.txt'), delimiter=' ')
    y_test = np.loadtxt(os.path.normpath('in/Test/y_test.txt'))
    X_train = np.loadtxt(os.path.normpath('in/Train/X_train.txt'), delimiter=' ')
    y_train = np.loadtxt(os.path.normpath('in/Train/y_train.txt'))

def buildPredictiveModel(type):
    if type == 'SVM':
        return svm.SVC().fit(X_train, y_train).predict(X_test)
    elif type == 'kNN':
        return KNeighborsClassifier().fit(X_train, y_train).predict(X_test)
    elif type == 'DT':
        return DecisionTreeClassifier().fit(X_train, y_train).predict(X_test)
    elif type == 'RF':
        return RandomForestClassifier().fit(X_train, y_train).predict(X_test)


if __name__ == '__main__':
    getDataset()
    print('Model SVM:')
    print(buildPredictiveModel('SVM'))
    print('Model kNN:')
    print(buildPredictiveModel('kNN'))
    print('Model Decision Tree:')
    print(buildPredictiveModel('DT'))
    print('Model Random Forest:')
    print(buildPredictiveModel('RF'))
