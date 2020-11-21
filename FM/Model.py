import tensorflow as tf
from FM.Train import GradientDescent


class FM(tf.keras.Model):
    def __init__(self, input, K):
        super(FM, self).__init__()
        p = input.shape[1]

        # initialize model parameters
        self.w_0 = tf.Variable([0.], name='bias')
        self.w = tf.Variable(tf.zeros([p, 1]), name='linear_term')
        self.V = tf.Variable(tf.random.normal(shape=(p, K)), name='inter_term')

    def fit(self, X, y, optimizer, epochs, lambda_w=0, lambda_v=0, verbose=1):
        loss_values, y_hat = GradientDescent(self, X, y, optimizer, epochs, lambda_w, lambda_v, verbose)
        return loss_values, y_hat

    def get_linear_term(self, X):
        return tf.matmul(X, self.w)

    def get_inter_term(self, X):
        inter_term = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, self.V)) - \
                            tf.matmul(tf.square(X), tf.square( self.V)),
                            axis=1, keepdims=True)
        return inter_term

    def pred(self, X):
        linear_term = self.get_linear_term(X)
        inter_term = self.get_inter_term(X)
        y_hat = tf.add(self.w_0 , linear_term , inter_term)
        return y_hat

