import pandas as pd

if __name__ == '__main__':

    print(
        '------------------------------------------------Analiza X_test------------------------------------------------')
    X_test = pd.read_csv('out/X_test.csv')
    print('------------Ilość wartości------------')
    print(X_test.count())
    print('------------Ilość wartości unikalnych------------')
    print(X_test.nunique())
    print('------------Wartość średnia w zbiorze------------')
    print(X_test.mean())
    print('------------Ilość wartości null------------')
    print(X_test.isnull().sum())
    print('------------Wartość maksymalna------------')
    print(X_test.max())
    print('------------Wartość minimalna------------')
    print(X_test.min())
    print('------------Wartość najczęściej występująca w zbiorze------------')
    for col in X_test.columns:
        print(col)
        print(X_test[col].mode().tolist())

    print(
        '------------------------------------------------Analiza X_train------------------------------------------------')
    X_train = pd.read_csv('out/X_train.csv')
    print('------------Ilość wartości------------')
    print(X_train.count())
    print('------------Ilość wartości unikalnych------------')
    print(X_train.nunique())
    print('------------Wartość średnia w zbiorze------------')
    print(X_train.mean())
    print('------------Ilość wartości null------------')
    print(X_train.isnull().sum())
    print('------------Wartość maksymalna------------')
    print(X_train.max())
    print('------------Wartość minimalna------------')
    print(X_train.min())
    print('------------Wartość najczęściej występująca w zbiorze------------')
    for col in X_train.columns:
        print(col)
        print(X_train[col].mode().tolist())
