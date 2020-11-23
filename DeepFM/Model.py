import numpy as np
import tensorflow as tf

class FM(tf.keras.layers.Layer):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        self.num_inputs = int(sum(field_dims))  # d = sum(m_i)
        self.fc = tf.keras.layers.Embedding(self.num_inputs, 1)
        self.embedding = tf.keras.layers.Embedding(self.num_inputs, num_factors)
        self.linear_layer = tf.keras.layers.Dense(units=1, use_bias=True)

    def call(self, X):
        square_of_sum = np.sum(self.embedding(X), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(X) ** 2, axis=1)
        inter_term = np.sum((square_of_sum - sum_of_square), axis=1, keepdims=True)
        linear_term = np.sum(self.linear_layer(self.fc(X)))
        x = linear_term + 0.5 * inter_term
        return x


class DeepFM(tf.keras.layers.Layer):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()

        self.fm = FM(field_dims, num_factors)
        self.embedding = self.fm.embedding
        self.embed_output_dim = len(field_dims) * num_factors

        self.mlp = tf.keras.Sequential()
        for dim in mlp_dims:
            self.mlp.add(tf.keras.layers.Dense(dim, 'relu', True))
            self.mlp.add(tf.keras.layers.Dropout(rate=drop_rate))
        self.mlp.add(tf.keras.layers.Dense(units=1))

    def call(self, x):
        y_fm = self.fm(x)
        embed_x = self.embedding(x)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        y = tf.sigmoid(y_fm + self.mlp(inputs))
        return y