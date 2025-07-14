import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Dataset IV の生成
# -------------------------------
np.random.seed(42)
n = 200
d = 4
X = 3 * (np.random.rand(n, d) - 0.5)
X = np.hstack([X, np.ones((n, 1))])  # バイアス項（定数1）追加
d += 1
y = (2 * X[:, 0] - 1 * X[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y.astype(int) - 1

# -------------------------------
# 目的関数
# -------------------------------
def J(w, X, y, lambd):
    z = y * (X @ w)
    loss = np.log(1 + np.exp(-z)).sum()
    reg = (lambd / 2) * np.dot(w, w)
    return loss + reg

# -------------------------------
# 最急降下法（Fixed learning rate）
# -------------------------------
def steepest_gradient(X, y, lambd=1.0, lr=0.01, max_iter=100):
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(max_iter):
        z = y * (X @ w)
        grad = -((y[:, None] * X) / (1 + np.exp(z)[:, None])).sum(axis=0) + lambd * w
        w -= lr * grad
        losses.append(J(w, X, y, lambd))

    return w, np.array(losses)

# -------------------------------
# ニュートン法
# -------------------------------
def newton_method(X, y, lambd=1.0, max_iter=100):
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(max_iter):
        z = X @ w
        s = 1 / (1 + np.exp(-y * z))
        R = np.diag(s * (1 - s))
        grad = -((y[:, None] * X) / (1 + np.exp(y * z)[:, None])).sum(axis=0) + lambd * w
        H = X.T @ R @ X + lambd * np.eye(X.shape[1])
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w -= delta
        losses.append(J(w, X, y, lambd))

    return w, np.array(losses)

# -------------------------------
# 実行・比較
# -------------------------------
lambd = 1.0
max_iter = 100

w_sgd, loss_sgd = steepest_gradient(X, y, lambd=lambd, lr=0.1, max_iter=max_iter)
w_newton, loss_newton = newton_method(X, y, lambd=lambd, max_iter=max_iter)

J_star = min(loss_sgd[-1], loss_newton[-1])  # 比較用の最小値

# -------------------------------
# プロット
# -------------------------------
plt.figure(figsize=(8, 5))
plt.semilogy(np.abs(loss_sgd - J_star), label='Steepest Gradient')
plt.semilogy(np.abs(loss_newton - J_star), label='Newton Method')
plt.xlabel("Iteration")
plt.ylabel(r"$|J(w^{(t)}) - J(\hat{w})|$")
plt.title("Convergence of Optimization Methods")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
