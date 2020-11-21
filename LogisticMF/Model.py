import numpy as np
from LogisticMF.Train import ALS
from LogisticMF.Recommend import recommend_top_k
from utils.common.constants import DEFAULT_TOP_K

class LogisticMF():
    def __init__(self, R, f, alpha, lam):
        self.R = R
        self.f = f
        self.alpha = alpha
        self.lam = lam
        self.P = None
        self.n, self.m = R.shape

        # initialize
        self.X = np.random.normal(size=(self.n, self.f))
        self.Y = np.random.normal(size=(self.m, self.f))
        self.bias_u = np.random.normal(size=(self.n, 1))
        self.bias_i = np.random.normal(size=(1, self.m))

    def posterior_prob(self):
        # calculate log posterior (objective function)
        # our goal is to find parameters(X, Y, biases) that maximize it
        A = np.matmul(self.X, self.Y.T) + self.bias_u + self.bias_i
        exp_term = np.exp(A)
        posterior = np.sum(self.alpha * self.R * A)
        posterior -= np.sum((1 + self.alpha * self.R) * np.log(1 + exp_term))
        posterior -= 0.5 * self.lam * np.sum(np.square(self.X))
        posterior -= 0.5 * self.lam * np.sum(np.square(self.Y))
        return posterior

    def get_like_prob_mat(self):
        # calculate sigmoid(x_iy_i^T + bias_u + bias_i)
        exp_term = np.exp(np.matmul(self.X, self.Y.T) + self.bias_u + self.bias_i)
        self.P = np.divide(exp_term, 1 + exp_term)
        return self.P

    def get_gradients(self, fix_user=True):
        self.P = self.get_like_prob_mat()

        if fix_user:
            latent_factor_grad = np.matmul((self.alpha * self.R).T, self.X) \
                                 - np.matmul(((1 + self.alpha * self.R) * self.P).T, self.X) \
                                 - self.lam * self.Y
            bias_grad = np.sum(self.alpha * self.R - (1 + self.alpha * self.R) * self.P, axis=0)

        else:
            latent_factor_grad = np.matmul(self.alpha * self.R, self.Y) \
                                 - np.matmul((1 + self.alpha * self.R) * self.P, self.Y) - self.lam * self.X
            bias_grad = np.sum(self.alpha * self.R - (1 + self.alpha * self.R) * self.P, axis=1)

        return latent_factor_grad, bias_grad

    def fit(self, lr, epoch, verbose=1):
        posterior_values = ALS(self, lr, epoch, verbose)
        return posterior_values

    def recommend(self, top_k=DEFAULT_TOP_K):
        self.get_like_prob_mat()
        recommend_rst = recommend_top_k(self.R, self.P, top_k)
        return recommend_rst