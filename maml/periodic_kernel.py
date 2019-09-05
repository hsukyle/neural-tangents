from data import sinusoid_task
import ipdb
import numpy as onp
from sklearn import gaussian_process
import matplotlib.pyplot as plt

task = sinusoid_task(n_support=100, n_query=10, noise_std=0.05)

periodic_kernel = gaussian_process.kernels.ExpSineSquared()
gpr = gaussian_process.GaussianProcessRegressor(kernel=periodic_kernel,
                                                n_restarts_optimizer=100,
                                                alpha=0.05)


def assess(gpr, task):
    y_pred = gpr.predict(X=task['x_test'])
    print(f"test MSE: {onp.mean(onp.square(y_pred - task['y_test']))}")


def get_evals(kernel, task):
    k_train = kernel(task['x_train'])
    k_test = kernel(task['x_test'])
    evals_train, _ = onp.linalg.eigh(k_train)
    evals_test, _ = onp.linalg.eigh(k_test)

    return evals_train, evals_test


assess(gpr, task)
evals_train_pre, evals_test_pre = get_evals(periodic_kernel, task)
gpr.fit(X=task['x_train'], y=task['y_train'])
print(f"log marginal likelihood: {gpr.log_marginal_likelihood()}")
assess(gpr, task)
evals_train_post, evals_test_post = get_evals(gpr.kernel_, task)

y_pred_test = gpr.predict(X=task['x_test'])
plt.plot(task['x_test'], task['y_test'], 'kx', task['x_test'], y_pred_test, 'bo')
plt.savefig('periodic_kernel_regression_test.png')
plt.close()
y_pred_train = gpr.predict(X=task['x_train'])
plt.plot(task['x_train'], task['y_train'], 'kx', task['x_train'], y_pred_train, 'bo')
plt.savefig('periodic_kernel_regression_train.png')
plt.close()
plt.plot(onp.arange(len(evals_train_pre)), evals_train_pre, 'bo',
         onp.arange(len(evals_train_post)), evals_train_post, 'rx')
plt.legend('pre', 'post')
plt.savefig('periodic_kernel_evals_train.png')
plt.close()
plt.plot(onp.arange(len(evals_test_pre)), evals_test_pre, 'bo',
         onp.arange(len(evals_test_post)), evals_test_post, 'rx')
plt.legend('pre', 'post')
plt.savefig('periodic_kernel_evals_test.png')
plt.close()

print(f'optimized kernel periodicity: {gpr.kernel_.periodicity}')
print(f'optimized kernel lengthscale: {gpr.kernel_.length_scale}')
ipdb.set_trace()
x = 1
