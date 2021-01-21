from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

    X_test = pd.read_csv(filepath_or_buffer='in/Test/X_test.txt', header=None, sep=' ')
    y_test = pd.read_csv(filepath_or_buffer='in/Test/y_test.txt', header=None, sep=' ')
    X_train = pd.read_csv(filepath_or_buffer='in/Train/X_train.txt', header=None, sep=' ')
    y_train = pd.read_csv(filepath_or_buffer='in/Train/y_train.txt', header=None, sep=' ')


def buildClassifiers(x, x_pca, fitMode=True):
    result = {'SVC': buildClassifier(typeClassifier='SVC', x_pca=x_pca, fitMode=fitMode, weight=0.3),
              'kNN': buildClassifier(typeClassifier='kNN', x_pca=x_pca, fitMode=fitMode, weight=0.3),
              'DT': buildClassifier(typeClassifier='DT', x_pca=x_pca, fitMode=fitMode, weight=0.2),
              'RF': buildClassifier(typeClassifier='RF', x_pca=x_pca, fitMode=fitMode, weight=0.2)}
    return result


def buildClassifier(typeClassifier, x_pca, fitMode, weight):
    classifier = '';
    if typeClassifier == 'SVC':
        classifier = svm.SVC(C=5000, kernel='poly', degree=2)
    elif typeClassifier == 'kNN':
        classifier = RandomForestClassifier()
    elif typeClassifier == 'DT':
        classifier = KNeighborsClassifier()
    elif typeClassifier == 'RF':
        classifier = DecisionTreeClassifier()
    classifier.weight = weight
    if fitMode:
        classifier.fit(x_pca, y_train.values.ravel())

    return classifier


def buildPredictiveModel(classifiers, x):
    result = {}
    for classifier in classifiers:
        result[classifier] = classifiers[classifier].predict(x)
    return result


def calcEnsemblePredictions(predictions, classifiers, y):
    resultPredictions = []
    for i in range(len(list(predictions.values())[0])):
        votes = {}
        for classifier in classifiers:
            if predictions[classifier][i] in votes:
                votes[predictions[classifier][i]] \
                    += classifiers[classifier].weight
            else:
                votes[predictions[classifier][i]] \
                    = classifiers[classifier].weight
        votes = sorted(list(zip(votes.keys(), votes.values())), key=lambda v: v[1], reverse=True)
        if len(votes) == 2 and votes[0][1] == 0.5:
            if dict(y.value_counts())[(votes[0][0],)] > dict(y.value_counts())[(votes[1][0],)]:
                resultPredictions.append(votes[0][0])
            else:
                resultPredictions.append(votes[1][0])
        else:
            resultPredictions.append(votes[0][0])
    return resultPredictions


def rocauc(y,prediction):
    lb = LabelBinarizer()
    lb.fit(y)
    y_a = lb.transform(y)
    prediction = lb.transform(prediction)
    return roc_auc_score(y_a, prediction, average='macro')


def printStatistics(y, ensemblePredictions):
    print('Acc: {}'.format(accuracy_score(y, ensemblePredictions)))
    print('Recall: {}'.format(recall_score(y, ensemblePredictions, average='micro')))
    print("AUC  score: {}".format(rocauc(y,ensemblePredictions)))
    print('F1: {}'.format(f1_score(y, ensemblePredictions, average='micro')))


def createPrincipalComponentAnalysis():
    pca = PCA(n_components=100)
    x_train_pca = pca.fit_transform(X_train)
    x_test_pca = pca.transform(X_test)
    return x_train_pca, x_test_pca


def createPlots():
    markers = ['o', 'd', ',', 'x', '+', 'v', '^', '<', '>', '.', 's', 'h', 'H', 'D', 'd', 'P', 'X']
    colors = ['blue', 'red', 'yellow', 'cyan', 'green', 'purple', 'pink', 'black', 'peru', 'lime', 'grey', 'orange',
              'brown']

    X_train['class'] = y_train
    X_test['class'] = y_test

    col_a, col_b = 50, 200
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

    for i in set(X_train['class']):
        i = int(i)
        pred = X_train['class'] == i
        axes[0, 0].scatter(X_train[X_train.columns[col_a]][pred], X_train[X_train.columns[col_b]][pred], c=colors[i],
                           marker=markers[i], label='Class {}'.format(i))
    axes[0, 0].set_title('Train')
    axes[0, 0].legend()

    for i in set(X_test['class']):
        i = int(i)
        pred = X_test['class'] == i
        axes[0, 1].scatter(X_test[X_test.columns[col_a]][pred], X_test[X_test.columns[col_b]][pred], c=colors[i],
                           marker=markers[i], label='Class {}'.format(i))
    axes[0, 1].set_title('Test')
    axes[0, 1].legend()

    X_train['class'] = ensemblePredictionsTrain
    X_test['class'] = ensemblePredictionsTest

    for i in set(X_train['class']):
        i = int(i)
        pred = X_train['class'] == i
        axes[1, 0].scatter(X_train[X_train.columns[col_a]][pred], X_train[X_train.columns[col_b]][pred], c=colors[i],
                           marker=markers[i], label='Class {}'.format(i))
    axes[1, 0].set_title('Ensemble Predictions Train')
    axes[1, 0].legend()

    for i in set(X_test['class']):
        i = int(i)
        pred = X_test['class'] == i
        axes[1, 1].scatter(X_test[X_test.columns[col_a]][pred], X_test[X_test.columns[col_b]][pred], c=colors[i],
                           marker=markers[i], label='Class {}'.format(i))
    axes[1, 1].set_title('Ensemble Predictions Test')
    axes[1, 1].legend()
    fig = plt.gcf()
    fig.savefig('out/resultLab4_4.png')


if __name__ == '__main__':
    getDataset()

    x_train_pca, x_test_pca = createPrincipalComponentAnalysis()

    classifiersTrain = buildClassifiers(x=X_train, x_pca=x_train_pca)
    classifiersTest = buildClassifiers(x=X_test, x_pca=x_test_pca, fitMode=False)

    predictionsTrain = buildPredictiveModel(classifiers=classifiersTrain, x=x_train_pca)
    predictionsTest = buildPredictiveModel(classifiers=classifiersTrain, x=x_test_pca)

    ensemblePredictionsTrain = calcEnsemblePredictions(predictions=predictionsTrain, classifiers=classifiersTrain,
                                                       y=y_train)
    ensemblePredictionsTest = calcEnsemblePredictions(predictions=predictionsTest, classifiers=classifiersTest,
                                                      y=y_test)
    print('Statistics Test:')
    printStatistics(y=y_test, ensemblePredictions=ensemblePredictionsTest)
    print('Statistics Train:')
    printStatistics(y=y_train, ensemblePredictions=ensemblePredictionsTrain)

    createPlots()
