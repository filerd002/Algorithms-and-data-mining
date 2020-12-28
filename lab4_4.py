import numpy as np
import os

from sklearn import svm
from sklearn.model_selection import cross_val_score
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

def buildClassifiers(type):
    if type == 'SVM':
        return svm.SVC().fit(X_train, y_train)
    elif type == 'kNN':
        return KNeighborsClassifier().fit(X_train, y_train)
    elif type == 'DT':
        return DecisionTreeClassifier().fit(X_train, y_train)
    elif type == 'RF':
        return RandomForestClassifier().fit(X_train, y_train)

def average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    getDataset()

    svmClassifier = buildClassifiers('SVM')
    knnClassifier = buildClassifiers('kNN')
    dtClassifier = buildClassifiers('DT')
    rfClassifier = buildClassifiers('RF')


    cvsSvm = cross_val_score(svmClassifier, X_train, y_train, cv=5)
    cvsKnn = cross_val_score(knnClassifier, X_train, y_train, cv=5)
    cvsDt = cross_val_score(dtClassifier, X_train, y_train, cv=5)
    cvsRf = cross_val_score(rfClassifier, X_train, y_train, cv=5)

    print('SVM:')
    print(cvsSvm)
    print('kNN:')
    print(cvsKnn)
    print('DT:')
    print(cvsDt)
    print('RF:')
    print(cvsRf)

    print('---Wartości średniej z klasyfikacji---')
    print('SVM:')
    print("%f" % (average(cvsSvm)))
    print('kNN:')
    print("%f" % (average(cvsKnn)))
    print('DT:')
    print("%f" % (average(cvsDt)))
    print('RF:')
    print("%f" % (average(cvsRf)))

    print('---Średnie odchylenie standardowe z wyników klasyfikacji---')
    print('SVM:')
    print("%f" % (np.std(cvsSvm)))
    print('kNN:')
    print("%f" % (np.std(cvsKnn)))
    print('DT:')
    print("%f" % (np.std(cvsDt)))
    print('RF:')
    print("%f" % (np.std(cvsRf)))