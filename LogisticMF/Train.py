import numpy as np


def ALS(model, lr, epoch, verbose=1):
    X_grad_sum = np.zeros((model.n, model.f))
    Y_grad_sum = np.zeros((model.m, model.f))

    bias_u_grad_sum = np.zeros((model.n, 1))
    bias_i_grad_sum = np.zeros((1, model.m))

    posterior_values = []
    for i in range(epoch):
        # fix user, update item
        Y_grad, bias_i_grad = model.get_gradients()
        Y_grad_sum += np.square(Y_grad)
        bias_i_grad_sum += np.square(bias_i_grad)

        vec_step_size = lr / np.sqrt(Y_grad_sum)
        bias_step_size = lr / np.sqrt(bias_i_grad_sum)

        model.Y += vec_step_size * Y_grad
        model.bias_i += bias_step_size * bias_i_grad

        # fix item, update user
        X_grad, bias_u_grad = model.get_gradients(False)
        bias_u_grad = np.expand_dims(bias_u_grad, axis=1)
        X_grad_sum += np.square(X_grad)
        bias_u_grad_sum += np.square(bias_u_grad)

        vec_step_size = lr / np.sqrt(X_grad_sum)
        bias_step_size = lr / np.sqrt(bias_u_grad_sum)

        model.X += vec_step_size * X_grad
        model.bias_u += bias_step_size * bias_u_grad

        posterior = model.posterior_prob()
        posterior_values.append(posterior)

        if (i == 0) | ((i+1) % verbose == 0):
                print('step= {}, log posterior= {:.5f}'.format(i+1, posterior))

    return posterior_values

