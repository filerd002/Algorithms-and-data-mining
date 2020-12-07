import urllib.request
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def getDataset():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    raw_data = urllib.request.urlopen(url)
    dataset = pd.read_csv(raw_data)
    return dataset

if __name__ == '__main__':
    dataset = getDataset()
    X = dataset.iloc[:, 1:53]
    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(scaled_X)

    ax1.scatter(scaled_X[y_km == 0]['W0'], scaled_X[y_km == 0]['W1'], c='lightblue', marker='o', s=40, label='cluster 1')
    ax1.scatter(scaled_X[y_km == 1]['W0'], scaled_X[y_km == 1]['W1'], c='red', marker='s', s=40, label='cluster 2')
    ax1.set_title('K-means clustering')
    ax1.legend()
    ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                 linkage='complete')
    y_ac = ac.fit_predict(scaled_X)
    ax2.scatter(scaled_X[y_ac == 0]['W0'], scaled_X[y_ac == 0]['W1'], c='lightblue', marker='o', s=40, label='cluster 1')
    ax2.scatter(scaled_X[y_ac == 1]['W0'], scaled_X[y_ac == 1]['W1'], c='red', marker='s', s=40, label='cluster 2')
    ax2.set_title('Agglomerative clustering')
    ax2.legend()
    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(scaled_X)
    ax3.scatter(scaled_X[y_db == 0]['W0'], scaled_X[y_db == 0]['W1'], c='lightblue', marker='o', s=40, label='cluster 1')
    ax3.scatter(scaled_X[y_db == 1]['W0'], scaled_X[y_db == 1]['W1'], c='red', marker='s', s=40, label='cluster 2')
    ax3.set_title('DBSCAN clustering')
    ax3.legend()
    plt.savefig('out/analyse.png', dpi=300)



