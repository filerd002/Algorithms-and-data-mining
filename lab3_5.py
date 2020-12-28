import numpy as np
import os

from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

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


def rocauc(prediction, param):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y = lb.transform(y_test)
    prediction = lb.transform(prediction)
    score = roc_auc_score(y, prediction, average='macro')
    print("%s, AUC wynik: %s " % (param, score))
    return score


def analyse():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    print('---Analiza wstępna---')
    bestVal = 0
    bestKernel = ''
    for ker in kernels:
        svcData = svm.SVC(kernel=ker).fit(X_train, y_train)
        score = rocauc(svcData.predict(X_test), ker)
        if (score > bestVal):
            bestVal = score
            bestKernel = ker

    print("Najlepsza opcja to %s z wynikiem: %s" % (bestKernel, bestVal))

    for ker in kernels:
        c_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8138, 16276]
        print("---Analiza z parametrem C dla opcji %s:" % ker)
        bestVal = 0
        bestC = 1
        for val in c_values:
            svcData = svm.SVC(kernel=ker, C=val).fit(X_train, y_train)
            score = rocauc(svcData.predict(X_test), val)
            if (score > bestVal):
                bestVal = score
                bestC = val

        print("Opcja %s, najlepsza wartość parametru C to %s z wynikiem: %s" % (ker, bestC, bestVal))


if __name__ == '__main__':
    getDataset()
    analyse()
