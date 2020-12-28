import numpy as np
import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
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

def rocauc(prediction):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y = lb.transform(y_test)
    prediction = lb.transform(prediction)
    return roc_auc_score(y, prediction, average='macro')


def createConfusionMetrix(prediction, name):
    cm = confusion_matrix(y_test, prediction)
    df_cm = pd.DataFrame(cm, range(12), range(12))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
    fig = plt.gcf()
    plt.show()
    fig.savefig("out/%s_confusion_matrix.png" % name)

def evaluationEffectivenessClassification(svmPred, knnPred, dtPred, rfPred):
    print("ACC:")
    print("SVM %s" % accuracy_score(y_test, svmPred))
    print("kNN %s" % accuracy_score(y_test, knnPred))
    print("Decision Tree %s" % accuracy_score(y_test, dtPred))
    print("Random Forest %s" % accuracy_score(y_test, rfPred))

    print("RECALL:")
    print("SVM %s" % recall_score(y_test, svmPred, average='micro'))
    print("kNN %s" % recall_score(y_test, knnPred, average='micro'))
    print("Decision Tree %s" % recall_score(y_test, dtPred, average='micro'))
    print("Random Forest %s" % recall_score(y_test, rfPred, average='micro'))

    print("F1:")
    print("SVM %s" % f1_score(y_test, svmPred, average='micro'))
    print("kNN %s" % f1_score(y_test, knnPred, average='micro'))
    print("Decision Tree %s" % f1_score(y_test, dtPred, average='micro'))
    print("Random Forest %s" % f1_score(y_test, rfPred, average='micro'))

    print("AUC:")
    print("SVM %s" % rocauc(svmPred))
    print("kNN %s" % rocauc(knnPred))
    print("Decision Tree %s" % rocauc(dtPred))
    print("Random Forest %s" % rocauc(rfPred))


if __name__ == '__main__':
    getDataset()

    svmPred = buildPredictiveModel('SVM')
    createConfusionMetrix(svmPred, 'SVM')

    knnPred = buildPredictiveModel('kNN')
    createConfusionMetrix(knnPred, 'kNN')

    dtPred = buildPredictiveModel('DT')
    createConfusionMetrix(dtPred, 'DT')

    rfPred = buildPredictiveModel('RF')
    createConfusionMetrix(rfPred, 'RF')

    evaluationEffectivenessClassification(svmPred, knnPred, dtPred, rfPred)