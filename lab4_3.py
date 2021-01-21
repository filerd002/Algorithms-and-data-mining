from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

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

def createPrincipalComponentAnalysis():
    pca = PCA(n_components=100)
    x_train_pca = pca.fit_transform(X_train)
    x_test_pca = pca.transform(X_test)
    return x_train_pca, x_test_pca


def saveEEnsamblePredictionsFile(data):
    ensemblePredictions_df = pd.DataFrame({'class': data})
    print('Ensemble Predictions:')
    print(ensemblePredictions_df)
    ensemblePredictions_df.to_csv('out/ensambled_learning.csv', index=False)


if __name__ == '__main__':
    getDataset()

    x_train_pca, x_test_pca = createPrincipalComponentAnalysis()

    classifiersTrain = buildClassifiers(x=X_train, x_pca=x_train_pca)
    classifiersTest = buildClassifiers(x=X_test, x_pca=x_test_pca, fitMode=False)

    predictionsTrain = buildPredictiveModel(classifiers=classifiersTrain, x=x_train_pca)
    predictionsTest = buildPredictiveModel(classifiers=classifiersTrain, x=x_test_pca)

    ensemblePredictions = calcEnsemblePredictions(predictions=predictionsTrain, classifiers=classifiersTrain, y=y_train)
    ensemblePredictionsTest = calcEnsemblePredictions(predictions=predictionsTest, classifiers=classifiersTest, y=y_test)

    saveEEnsamblePredictionsFile(data=ensemblePredictions)
