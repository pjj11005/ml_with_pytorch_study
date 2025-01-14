import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from networks.logisticregressiongd import LogisticRegressionGD
from utils.getModules import load_iris_data, load_xor_data
from utils.tools import plot_decision_regions, plot_sigmoid, plot_logistic_loss_function, plot_xor_data, plot_entropy, plot_impurity


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


# SVM
def train_svm(X_train_std, y_train, X_combined_std, y_combined):
    # train
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_std, y_train)

    # decision boundary
    plot_decision_regions(X_combined_std, y_combined, classifier=svm,
                        test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_11.png', dpi=300)
    plt.show()


# kernel SVM
def train_kernel_svm(X_xor, y_xor, X_train_std, y_train, X_combined_std, y_combined):
    # train
    svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)

    # decision boundary
    plot_decision_regions(X_xor, y_xor,
                        classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_14.png', dpi=300)
    plt.show()

    # train(gamma=0.2)
    svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)

    # decision boundary
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_15.png', dpi=300)
    plt.show()

    # train(gamma=100.0) -> 규제 완화
    svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
    svm.fit(X_train_std, y_train)

    # decision boundary
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_16.png', dpi=300)
    plt.show()


# DecisionTree
def train_dt(X_train, y_train, X_combined, y_combined):
    # train
    tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)
    tree_model.fit(X_train, y_train)

    # decision boundary
    plot_decision_regions(X_combined, y_combined,
                      classifier=tree_model,
                      test_idx=range(105, 150))
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_20.png', dpi=300)
    plt.show()

    # tree structure
    feature_names = ['Petal length', 'Petal width']
    tree.plot_tree(tree_model,
                feature_names=feature_names,
                filled=True)
    # plt.savefig('figures/03_21.pdf')
    plt.show()


# RandomForest
def train_rf(X_train, y_train, X_combined, y_combined):
    # train
    forest = RandomForestClassifier(n_estimators=25,
                                random_state=1,
                                n_jobs=2)
    forest.fit(X_train, y_train)

    # decision boundary
    plot_decision_regions(X_combined, y_combined,
                        classifier=forest, test_idx=range(105, 150))
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_22.png', dpi=300)
    plt.show()


# KNN
def train_knn(X_train_std, y_train, X_combined_std, y_combined):
    # train
    knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
    knn.fit(X_train_std, y_train)

    # decision boundary
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=knn, test_idx=range(105, 150))
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('figures/03_24.png', dpi=300)
    plt.show()


def main():
    # data load
    X_train, X_test, y_train, y_test, X_train_std, X_test_std = load_iris_data()

    # data combine
    X_combined_std = np.vstack((X_train_std, X_test_std))
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # class 0, class 1 -> subset
    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    # xor data load
    X_xor, y_xor = load_xor_data()

    '''train Perceptron'''
    # train_ppn(X_train_std, X_test_std, y_train, y_test, X_combined_std, y_combined)

    '''plot sigmoid function'''
    # plot_sigmoid()

    '''plot logistic loss function'''
    # plot_logistic_loss_function()

    '''train LogisticRegressionGD'''
    # train_lrgd(X_train_01_subset, y_train_01_subset)

    '''train LogisticRegression'''
    # train_lr(X_train_std, X_test_std, y_train, X_combined_std, y_combined)

    '''train SVM'''
    # train_svm(X_train_std, y_train, X_combined_std, y_combined)

    '''plot xor data'''
    # plot_xor_data()

    '''train kernel SVM'''
    # train_kernel_svm(X_xor, y_xor, X_train_std, y_train, X_combined_std, y_combined)

    '''plot entropy'''
    # plot_entropy()

    '''plot impurity'''
    # plot_impurity()

    '''train DecisionTree'''
    # train_dt(X_train, y_train, X_combined, y_combined)

    '''train RandomForest'''
    # train_rf(X_train, y_train, X_combined, y_combined)

    '''train KNN'''
    train_knn(X_train_std, y_train, X_combined_std, y_combined)


if __name__ == '__main__':
    main()
