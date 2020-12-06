import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import urllib.request as urllib
import matplotlib.pyplot as plt

def getDataset():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    raw_data = urllib.urlopen(url)
    dataset = pd.read_csv(raw_data)
    return dataset


def plotKMeans( scaled_X, clasters = 5):
    markers = ['o', 'd', ',', 'x', '+', 'v', '^', '<', '>', '.', 's']
    colors = []
    for x in range(0, clasters):
        colors.append(np.random.rand(3, ))
    while len(markers) < clasters:
        clasters.append('o')


    kmeansnPlus = KMeans(n_clusters=clasters, init='k-means++')

    kmeans = KMeans(n_clusters=clasters, init='random')

    y_kmeansPlus = kmeansnPlus.fit_predict(scaled_X)
    y_kmeans = kmeans.fit_predict(scaled_X)

    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

    for x in range(0, clasters):
        axes[0].scatter(scaled_X[y_kmeans == x]['W0'], scaled_X[y_kmeans == x]['W1'], c=[colors[x]], marker=markers[x], label='Skupienie ' + str(x +1))
    axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black',marker="*", label='Centroidy')
    axes[0].set_title('KMeans')
    axes[0].set_xlabel('Pierwszy tydzień')
    axes[0].set_ylabel('Drugi tydzień')
    axes[0].legend()

    for x in range(0, clasters):
        axes[1].scatter(scaled_X[y_kmeansPlus == x]['W0'], scaled_X[y_kmeansPlus == x]['W1'], c=[colors[x]], marker=markers[x], label='Skupienie ' + str(x +1))
    axes[1].scatter(kmeansnPlus.cluster_centers_[:, 0], kmeansnPlus.cluster_centers_[:, 1], c='black',marker="*", label='Centroidy')
    axes[1].set_title('KMeans++')
    axes[1].set_xlabel('Pierwszy tydzień')
    axes[1].set_ylabel('Drugi tydzień')
    axes[1].legend()

    fig = plt.gcf()
    plt.show()
    fig.savefig('out/KMeans.png')




if __name__ == '__main__':
    dataset = getDataset()
    print (dataset)
    X = dataset.iloc[:, 1:53]
    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns
    plotKMeans(scaled_X, 5)



