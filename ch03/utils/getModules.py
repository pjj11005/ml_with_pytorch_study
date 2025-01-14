from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_iris_data():
    # 데이터 로드
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    
    # 스케일링
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_std, X_test_std


# xor data 
def load_xor_data():
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor((X_xor[:, 0] > 0),
                        X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)

    return X_xor, y_xor
