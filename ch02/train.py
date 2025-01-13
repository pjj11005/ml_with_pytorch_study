import numpy as np
import matplotlib.pyplot as plt

from utils.getModules import load_iris_data
from utils.tools import scatter_iris, plot_decision_regions
from networks.perceptron import Perceptron
from networks.adalinegd import AdalineGD
from networks.adalinesgd import AdalineSGD


# Perceptron
def train_perceptron(X, y):
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # 가중치 업데이트 시각화
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    # plt.savefig('images/02_07.png', dpi=300)
    plt.show()

    # 결정경계 그래프
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    # plt.savefig('images/02_08.png', dpi=300)
    plt.show()


# AdalineGD
def train_adalinegd(X, y, X_std):
    # 학습률에 따른 loss 그래프
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epoches')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate 0.1')

    ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    # plt.savefig('images/02_11.png', dpi=300)
    plt.show()

    # 특성 스케일 조정라여 경사 하강법 결과 향상
    ada_gd = AdalineGD(n_iter=20, eta=0.5)
    ada_gd.fit(X_std, y)
    # 결정 경계
    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient descent')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('images/02_14_1.png', dpi=300)
    plt.show()
    # MSE / Epochs
    plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.tight_layout()
    # plt.savefig('images/02_14_2.png', dpi=300)
    plt.show()

# AdalineSGD
def train_adalinesgd(X, y, X_std):
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plt.title('Adaline - Stochastic gradient descent')
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('images/02_15_1.png', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average loss')

    # plt.savefig('images/02_15_2.png', dpi=300)
    plt.show()

    # partial fit
    # ada_sgd.partial_fit(X_std[0, :], y[0])



# 데이터 로드
X, y, X_std = load_iris_data()

# iris 산점도 시각화
# scatter_iris(X)

# train Perceptron 
# train_perceptron(X, y)

# train AdalineGD 
# train_adalinegd(X, y, X_std)

# train AdalineSGD 
train_adalinesgd(X, y, X_std)
