import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.getModules import load_iris_data


# iris data 산점도
def scatter_iris(X):
    # 산점도
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')

    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    # plt.savefig('images/02_06.png', dpi=300)
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 마커와 컬러맵을 설정
    markers = ('o', 's', '^', 'v', '<')
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # ravel() : 1차원으로 펼침
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 샘플의 산점도를 그리기
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=[colors[idx]],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
