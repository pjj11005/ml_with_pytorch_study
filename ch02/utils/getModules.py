import os
import pandas as pd
import numpy as np


def load_iris_data():
    try:
        s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        print('From URL', s)
        df = pd.read_csv(s, header=None, encoding='utf-8')

    except HTTPError:
        s = 'iris.data'
        print('From local iris path:', s)
        df = pd.read_csv(s, header=None, encoding='utf-8')

    # setosa와 vesicolor를 선택
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1) # setosa: 0, vesicolor: 1

    # 꽃받침 길이, 꽃잎 길이 추출
    X = df.iloc[0:100, [0, 2]].values

    # 특성 표준화
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    return X, y, X_std
