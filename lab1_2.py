import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def splitDataset(x, y):
    return train_test_split(x, y, test_size=0.4)

def saveToCSV(data, names, path):
    pd.DataFrame(data, columns=names).to_csv(path)

if __name__ == '__main__':
    dataset = load_wine()
    x = dataset.data
    y = dataset.target
    x_train, x_test, y_train, y_test = splitDataset(x, y)

    columns_label = ['target']

    saveToCSV(x_train, dataset.feature_names, "out/X_train.csv")
    saveToCSV(y_train, columns_label, "out/Y_train.csv")

    saveToCSV(x_test, dataset.feature_names, "out/X_test.csv")
    saveToCSV(y_test, columns_label, "out/Y_test.csv")
