import tensorflow as tf

def train(model, X, y, optimizer, epochs, lambda_w=0, lambda_v=0, verbose=1):
    loss_values = []
    for i in range(epochs):
        with tf.GradientTape() as tape:
            y_hat = model.pred(X)
            error = tf.reduce_mean(tf.square(y - y_hat))
            l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(model.w, 2)),
                                        tf.multiply(lambda_v, tf.pow(model.V, 2))))
            loss = tf.add(error, l2_norm)
            loss_values.append(loss)
        # get gradients
        grad = tape.gradient(loss, model.trainable_variables)
        # update
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # print loss
        if (i == 0) | ((i+1) % verbose == 0):
            print('step= {}, error= {:.5f}, loss= {:.5f}'.format(i+1, error, loss))

    return loss_values, y_hat
