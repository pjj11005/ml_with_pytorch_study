import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 맷플롯립 호환성 체크
import matplotlib
from distutils.version import LooseVersion


# decision regions
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # 마커와 컬러맵을 설정
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 결정 경계를 그림
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 클래스 샘플을 그림
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # 테스트 샘플을 부각하여 그림
    if test_idx:
        # 모든 샘플을 그림
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')


# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# plot sigmoid
def plot_sigmoid():
    z = np.arange(-7, 7, 0.1)
    sigma_z = sigmoid(z)

    plt.plot(z, sigma_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\sigma (z)$')

    # y 축의 눈금과 격자선
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    # plt.savefig('figures/03_02.png', dpi=300)
    plt.show()


# plot logistic loss function
def plot_logistic_loss_function():
    def loss_1(z):
        return - np.log(sigmoid(z))

    def loss_0(z):
        return - np.log(1 - sigmoid(z))

    z = np.arange(-10, 10, 0.1)
    sigma_z = sigmoid(z)

    c1 = [loss_1(x) for x in z]
    plt.plot(sigma_z, c1, label='L(w, b) if y=1')

    c0 = [loss_0(x) for x in z]
    plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')

    plt.ylim(0.0, 5.1)
    plt.xlim([0, 1])
    plt.xlabel('$\sigma(z)$')
    plt.ylabel('L(w, b)')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('figures/03_04.png', dpi=300)
    plt.show()

    
