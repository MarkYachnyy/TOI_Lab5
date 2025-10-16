import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def vknn_nd(x, XN, k):
    N = XN.shape[1]
    p_ = np.zeros(x.shape[1])
    for idx in range(x.shape[1]):
        dist = np.linalg.norm(XN - x[:, idx:idx+1], axis=0)
        sorted_dist = np.sort(dist)
        V = np.pi * sorted_dist[k-1] ** 2  # объем окрестности для двумерного случая (круг)
        p_[idx] = k / (N * V) if V > 0 else 0
    return p_

n = 2
N = 2000
gm = 0.5
k = round(N ** gm)

M = 3
ps = [0.2, 0.2, 0.6]
D = 0.2
ro = -np.log(0.7)
m1 = np.array([[0], [0]])
m2 = np.array([[1], [0]])
m3 = np.array([[0], [1]])
m = np.hstack([m1, m2, m3])
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        C[i, j] = D * np.exp(-ro * abs(i - j))

x1 = np.arange(-2, 3.1, 0.1)
x2 = np.arange(-2, 3.1, 0.1)
X1, X2 = np.meshgrid(x1, x2)
x_flat = np.vstack([X1.ravel(), X2.ravel()])

# Смесь гауссовских распределений
p = (
    ps[0] * multivariate_normal.pdf(x_flat.T, mean=m1.ravel(), cov=C) +
    ps[1] * multivariate_normal.pdf(x_flat.T, mean=m2.ravel(), cov=C) +
    ps[2] * multivariate_normal.pdf(x_flat.T, mean=m3.ravel(), cov=C)
)
pi = p.reshape(len(x1), len(x2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, pi, cmap='viridis')
ax.set_title('Распределение смеси гауссовских векторов', fontsize=14)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('pi')
plt.show()

XN = np.zeros((n, N))
for i in range(N):
    u = np.random.rand()
    if u < ps[0]:
        t = 0
    elif u < ps[0] + ps[1]:
        t = 1
    else:
        t = 2
    XN[:, i] = np.random.multivariate_normal(mean=m[:, t], cov=C)

p_ = vknn_nd(x_flat, XN, k)
pv = p_.reshape(len(x1), len(x2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, pv, cmap='viridis')
ax.set_title('Оценка плотности распределения смеси', fontsize=14)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('p~')
plt.show()

# Линии постоянного уровня
plt.figure()
contour1 = plt.contour(X1, X2, pi, levels=[0.001, 0.01, 0.5*np.max(pi)])
plt.clabel(contour1)
plt.title('Линии постоянного уровня плотности', fontsize=14)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.figure()
contour2 = plt.contour(X1, X2, pv, levels=[0.001, 0.01, 0.5*np.max(pv)])
plt.clabel(contour2)
plt.title('Линии постоянного уровня оценки плотности', fontsize=14)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
