import numpy as np
import os

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


if __name__ == '__main__':
    getDataset()
    print(X_test)
    print(y_test)
    print(X_train)
    print(y_train)
