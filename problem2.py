import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

# Soft-thresholding operator for L1 norm
def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

# Proximal Gradient Method
def proximal_gradient(A, mu, lambd, eta, max_iter=1000, tol=1e-8):
    w = np.zeros_like(mu)
    history = []

    for _ in range(max_iter):
        grad = A @ (w - mu)
        w_new = soft_thresholding(w - eta * grad, eta * lambd)
        loss = 0.5 * ((w_new - mu).T @ A @ (w_new - mu)) + lambd * np.sum(np.abs(w_new))
        history.append(loss)
        if np.linalg.norm(w - w_new) < tol:
            break
        w = w_new
    return w, history

# AdaGrad Method
def adagrad(A, mu, lambd, eta, max_iter=1000, tol=1e-8):
    w = np.zeros_like(mu)
    history = []
    G = np.zeros_like(w)

    for _ in range(max_iter):
        grad = A @ (w - mu)
        G += grad ** 2
        adjusted_eta = eta / (np.sqrt(G) + 1e-8)
        w_new = soft_thresholding(w - adjusted_eta * grad, adjusted_eta * lambd)
        loss = 0.5 * ((w_new - mu).T @ A @ (w_new - mu)) + lambd * np.sum(np.abs(w_new))
        history.append(loss)
        if np.linalg.norm(w - w_new) < tol:
            break
        w = w_new
    return w, history

# Question 1: Effect of lambda on w_hat
A1 = np.array([[3, 0.5], [0.5, 1]])
mu1 = np.array([1.0, 2.0])
L1 = max(eigvals(A1))
eta1 = 1 / L1

lambdas = np.arange(0.01, 10.01, 0.01)
w_solutions = []

for lam in lambdas:
    w_hat, _ = proximal_gradient(A1, mu1, lam, eta1)
    w_solutions.append(w_hat)

w_solutions = np.array(w_solutions)

# Plot 1: Effect of lambda on w_hat
plt.figure(figsize=(8, 5))
plt.plot(lambdas, w_solutions[:, 0], label='w[0]')
plt.plot(lambdas, w_solutions[:, 1], label='w[1]')
plt.xlabel("lambda")
plt.ylabel("w_hat components")
plt.title("Effect of lambda on w_hat")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Question 2: PG vs AdaGrad under ill-conditioned matrix
A2 = np.array([[300, 0.5], [0.5, 10]])
mu2 = np.array([1.0, 2.0])
L2 = max(eigvals(A2))
eta2 = 1 / L2
lambda2 = 1.0

w_pg, history_pg = proximal_gradient(A2, mu2, lambda2, eta2)
w_ada, history_ada = adagrad(A2, mu2, lambda2, eta2)

# Plot 2: Objective value vs iteration
plt.figure(figsize=(8, 5))
plt.semilogy(history_pg, label="Proximal Gradient")
plt.semilogy(history_ada, label="AdaGrad")
plt.xlabel("Iteration")
plt.ylabel("Objective value (log scale)")
plt.title("Comparison of PG and AdaGrad")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

