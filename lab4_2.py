import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time

global X_test
global y_test
global X_train
global y_train


def getDataset():
    global X_test
    global y_test
    global X_train
    global y_train

    X_test = pd.read_csv(filepath_or_buffer='in/Test/X_test.txt', header=None, sep=' ')
    y_test = pd.read_csv(filepath_or_buffer='in/Test/y_test.txt', header=None, sep=' ')
    X_train = pd.read_csv(filepath_or_buffer='in/Train/X_train.txt', header=None, sep=' ')
    y_train = pd.read_csv(filepath_or_buffer='in/Train/y_train.txt', header=None, sep=' ')




def kncClassifier(x_train, x_test, y_train, y_test, withPca):
    if (withPca):
        print("KN Classifier with PCA")
        pca = PCA(n_components=100)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    startClasifier = time.time()
    kncClass = KNeighborsClassifier().fit(x_train, y_train.values.ravel())
    endClassifier = time.time()

    trainingTime = endClassifier - startClasifier

    startPrediction = time.time()
    predict = kncClass.predict(x_test)
    endPrediction = time.time()
    testingTime = endPrediction - startPrediction
    accScore = accuracy_score(y_test, predict)

    print("knc ACC: %s" % accScore)

    cvsKnc = cross_val_score(kncClass, x_train, y_train.values.ravel(), cv=5)

    for score in cvsKnc:
        print("cvs: %f" % score)

    print("training time:  %f, testing time: %f" % (trainingTime, testingTime))

    return accScore, cvsKnc, trainingTime, testingTime


def prepareDataToDF(fistrResult, secountResult):
    data = []
    line = []
    line.append('csvScore')
    line.append('accScore')
    line.append('trainingTime')
    line.append('testingTime')
    data.append(line)
    for result in fistrResult[1]:
        newLine = []
        newLine.append(result)
        newLine.append(fistrResult[0])
        newLine.append(fistrResult[2])
        newLine.append(fistrResult[3])
        data.append(newLine)
    line = []
    line.append('PCA csvScore')
    line.append('PCA accScore')
    line.append('PCA trainingTime')
    line.append('PCA testingTime')
    data.append(line)
    for result in secountResult[1]:
        newLine = []
        newLine.append(result)
        newLine.append(secountResult[0])
        newLine.append(secountResult[2])
        newLine.append(secountResult[3])
        data.append(newLine)
    return data


def saveReductionFile(data):
    df = pd.DataFrame(data)
    print(df)
    writer = pd.ExcelWriter("out/dim_reduction.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='result', index=False, header=False)
    writer.save()

if __name__ == '__main__':
    getDataset()
    knnClas = kncClassifier(X_train, X_test, y_train, y_test, False)
    knnClasPCA = kncClassifier(X_train, X_test, y_train, y_test, True)
    data = prepareDataToDF(knnClas, knnClasPCA)
    saveReductionFile(data)

