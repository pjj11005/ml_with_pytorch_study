import numpy as np


class AdalineGD:
    '''적응형 선형 뉴런 분류기

    매개변수
    -----------
    eta : float
        학습률(0.0과 1.0 사이)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드

    속성
    ---------
    w_ : 1d-array
        학습된 가중치
    b_ : 스칼라
        학습된 절편 유닛
    errors_ : list
        에포크마다 누적된 분류 오류
    '''

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        '''훈련 데이터 학습

        매개변수
        ----------
        X : array-like, shape = [n_sample, n_features]
            n_samples개의 샘플과 n_features개의 특성으로 이루어진 훈련 데이터
        y : array-like, shape = [n_sample]
            타겟값

        반환값
        ---------
        self : object
        '''

        rgen = np.random.RandomState(self.random_state)
        self.w_  = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            '''
            활성화 함수 = 항등 함수(identify function)으로 아무런 효과 없음
            대신 'output = self.net_input(X)'로 바꿀 수 있음
            로지스틱 회귀의 경우 이 함수를 시그모이드 함수로 바꾸어 로지스틱 회귀 분류기를 구현함
            '''
            output = self.activation(net_input)
            errors = (y - output)

            '''
            for w_j in range(self.w_.shape[0]):
                self.w_[w_j] += self.eta * (2.0 * (X[:, w_j] * errors)).mean()
            '''

            # 전체 훈련 데이터셋에 기반으로 그레이디언트 계산
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        '''최종 입력계산'''
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """선형 활성화 계산"""
        return X

    def predict(self, X):
        '''단위 계단 함수를 사용하여 클래스 레이블 반환'''
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)