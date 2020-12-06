import pandas as pd
import seaborn as sns
from networkx.drawing.tests.test_pylab import plt
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Data set
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv'
    df = pd.read_csv(url)
    X = df.iloc[:, 1:53]
    scaler = StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X))
    scaled_X.columns = X.columns

    sns.clustermap(scaled_X, figsize=(50, 50))
    plt.savefig('out/hierarchical_clustered_heatmap.png', dpi=300)

    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(scaled_X)

    plt.scatter(scaled_X[y_db == 0, 0], scaled_X[y_db == 0, 1], c='lightblue', marker='o', s=40, label='cluster 1')
    plt.scatter(scaled_X[y_db == 1, 0], scaled_X[y_db == 1, 1], c='red', marker='s', s=40, label='cluster 2')

    plt.legend()
    plt.show()




