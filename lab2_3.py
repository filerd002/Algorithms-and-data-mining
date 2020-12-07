import urllib
import pandas as pd
import seaborn as sns
from networkx.drawing.tests.test_pylab import plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def getDataset():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv"
    raw_data = urllib.request.urlopen(url)
    dataset = pd.read_csv(raw_data)
    return dataset


def getResultAgglomerativeClustering(dataset, num_clusters=5, metric_name='euclidean'):
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity=metric_name, memory=None,
                                         connectivity=None, compute_full_tree='auto', linkage='ward',
                                         distance_threshold=None).fit(dataset)
    result = dataset.copy()
    result['TEST'] = list(ac.labels_)
    return result

def saveDendrogram(model):
    sns.clustermap(model, figsize=(50, 50))
    plt.savefig('out/hierarchical_clustered_heatmap.png', dpi=300)


if __name__ == '__main__':
    dataset = getDataset()
    X = dataset.iloc[:, 1:53]
    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns

    agglomerativeClustering_X = getResultAgglomerativeClustering(scaled_X, 5, 'euclidean')

    saveDendrogram(agglomerativeClustering_X);





