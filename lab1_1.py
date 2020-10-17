import pandas as pd

from sklearn.datasets import load_wine

if __name__ == '__main__':
    data = load_wine()
    print(load_wine())
    print(data.keys())
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.head()
    df.info();

