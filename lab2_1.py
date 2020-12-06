import pandas as pd
import urllib.request as urllib


if __name__ == '__main__':
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00396/Sales_Transactions_Dataset_Weekly.csv'
    raw_data = urllib.urlopen(url)
    dataset = pd.read_csv(raw_data)
    print(dataset)
    data_norm = dataset.copy()

    print(data_norm[['Normalized {}'.format(i) for i in range(0, 52)]].head())







