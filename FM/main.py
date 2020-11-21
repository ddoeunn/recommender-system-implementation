import numpy as np
import tensorflow as tf
from FM.Model import FM

# Data
X = np.array([
    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
], dtype=np.float32)
y = np.array([5, 3, 1, 4, 5, 1, 5], dtype=np.float32)
y = np.expand_dims(y, axis=1)

# Hyper Parameters
K = 5
lr = 0.1
epochs = 500
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)

# Model
model = FM(X, K)

# Training
loss_values, y_hat = model.fit(X, y, optimizer, epochs, lambda_w, lambda_v, verbose=50)

# Result
print(y_hat) # y_true = [5, 3, 1, 4, 5, 1, 5]
print(model.pred(X))