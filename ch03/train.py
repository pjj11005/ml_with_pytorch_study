import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from utils.getModules import load_iris_data
from utils.tools import plot_decision_regions, plot_sigmoid, plot_logistic_loss_function
from networks.logisticregressiongd import LogisticRegressionGD


# Perceptron
def train_ppn(X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined):
    # train
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    
    # predict, accuracy
    y_pred = ppn.predict(X_test_std)
    print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
    print('정확도: %.3f' % accuracy_score(y_test, y_pred))
    print('정확도: %.3f' % ppn.score(X_test_std, y_test))

    # ppn decision boundary
    plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_01.png', dpi=300)
    plt.show()


# LogisticRegressionGD 
def train_lrgd(X_train_01_subset, y_train_01_subset):
    # train
    lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    lrgd.fit(X_train_01_subset,
            y_train_01_subset)

    # decision boundary
    plot_decision_regions(X=X_train_01_subset,
                        y=y_train_01_subset,
                        classifier=lrgd)
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_05.png', dpi=300)
    plt.show()


# LogisticRegression
def train_lr(X_train_std, X_test_std, y_train, X_combined_std, y_combined, multiclass='ovr'):
    # train
    lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class=multiclass)
    lr.fit(X_train_std, y_train)

    # decision boundary
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=lr, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_06.png', dpi=300)
    plt.show()

    # predict_proba
    lr.predict_proba(X_test_std[:3, :])
    lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
    lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)

    # predict
    lr.predict(X_test_std[:3, :])
    lr.predict(X_test_std[0, :].reshape(1, -1))

    # 규제(C)를 사용하여 과대적합 피하기
    weights, params = [], []
    for c in np.arange(-5, 5):
        lr = LogisticRegression(C=10.**c,
                                multi_class='ovr')
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.**c)

    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='Petal length')
    plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
    plt.xlabel('C')
    plt.ylabel('Weight coefficient')
    plt.legend(loc='upper left')
    plt.xscale('log')
    # plt.savefig('figures/03_08.png', dpi=300)
    plt.show()

def main():
    # data load
    X_train, X_test, y_train, y_test, X_train_std, X_test_std = load_iris_data()

    # data combine
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # class 0, class 1 -> subset
    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    '''train Perceptron'''
    # train_ppn(X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined)

    '''plot sigmoid function'''
    # plot_sigmoid()

    '''plot logistic loss function'''
    # plot_logistic_loss_function()

    '''train LogisticRegressionGD'''
    # train_lrgd(X_train_01_subset, y_train_01_subset)

    '''train LogisticRegression'''
    train_lr(X_train_std, X_test_std, y_train, X_combined_std, y_combined)

if __name__ == '__main__':
    main()
