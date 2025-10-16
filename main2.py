import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def vknn_2d(x, XN, k):
    """Оценка плотности методом K ближайших соседей для двумерного случая"""
    N = XN.shape[1]
    p_ = np.zeros(x.shape[1])
    for idx in range(x.shape[1]):
        dist = np.linalg.norm(XN - x[:, idx:idx + 1], axis=0)
        sorted_dist = np.sort(dist)
        V = np.pi * sorted_dist[k - 1] ** 2  # объем окрестности (площадь круга)
        p_[idx] = k / (N * V) if V > 0 else 0
    return p_


# Задаем перебираемые значения величины g
GG = np.arange(0.1, 1.0, 0.1)
err = np.zeros_like(GG)  # массив значений ошибок

# Цикл по числу элементов GG
for tt in range(len(GG)):
    # 1. Исходные данные
    n = 2  # размерность вектора наблюдений
    N = 2000  # количество используемых для оценки векторов
    gm = GG[tt]  # подставляем очередное значение из массива GG
    k = round(N ** gm)  # k - число ближайших соседей

    # 2. Параметры двумерного гауссовского распределения
    mean = np.array([0, 0])  # математическое ожидание
    cov = np.array([[1, 0.5],  # ковариационная матрица
                    [0.5, 1]])

    # Генерация сетки для визуализации
    x1 = np.arange(-4, 4.1, 0.2)
    x2 = np.arange(-4, 4.1, 0.2)
    X1, X2 = np.meshgrid(x1, x2)
    x_flat = np.vstack([X1.ravel(), X2.ravel()])

    # Эталонная плотность (истинная плотность двумерной гауссовской величины)
    p = multivariate_normal.pdf(x_flat.T, mean=mean, cov=cov)

    # 3. Генерация обучающей выборки
    XN = np.random.multivariate_normal(mean, cov, N).T

    # 4. Оценка плотности по методу К ближайших соседей
    p_ = vknn_2d(x_flat, XN, k)

    # Фиксируем среднеквадратичную ошибку
    err[tt] = np.sqrt(np.mean((p - p_) ** 2))

    print(f'g = {gm:.1f}, k = {k}, ошибка = {err[tt]:.6f}')

# Вывод зависимости ошибки от величины g
plt.figure(figsize=(10, 6))
plt.plot(GG, err, '-o', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('g', fontsize=12)
plt.ylabel('Среднеквадратичная ошибка', fontsize=12)
plt.title('Зависимость ошибки оценивания от параметра g (двумерный случай)', fontsize=14)

# Находим оптимальное значение g
optimal_idx = np.argmin(err)
optimal_g = GG[optimal_idx]
optimal_err = err[optimal_idx]
optimal_k = round(N ** optimal_g)

plt.axvline(x=optimal_g, color='r', linestyle='--', label=f'Оптимальное g = {optimal_g:.1f}')
plt.legend(fontsize=11)
plt.text(optimal_g + 0.05, optimal_err, f'min ошибка = {optimal_err:.4f}',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

print(f'\nОптимальное значение g: {optimal_g:.1f}')
print(f'Соответствующее k: {optimal_k}')
print(f'Минимальная ошибка: {optimal_err:.6f}')

plt.tight_layout()
plt.show()

# Визуализация для оптимального значения g
print('\nВизуализация результатов для оптимального g...')

# Пересчитываем с оптимальным g
gm_opt = optimal_g
k_opt = round(N ** gm_opt)
XN_opt = np.random.multivariate_normal(mean, cov, N).T
p_true = multivariate_normal.pdf(x_flat.T, mean=mean, cov=cov)
p_est = vknn_2d(x_flat, XN_opt, k_opt)

# Преобразуем в матрицы для отображения
p_true_grid = p_true.reshape(len(x1), len(x2))
p_est_grid = p_est.reshape(len(x1), len(x2))

# График 1: Истинная плотность
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X1, X2, p_true_grid, cmap='viridis', alpha=0.8)
ax1.set_title(f'Истинная плотность\n(двумерная гауссовская)', fontsize=12)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('p')

# График 2: Оценка плотности
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X1, X2, p_est_grid, cmap='plasma', alpha=0.8)
ax2.set_title(f'Оценка плотности\n(g={gm_opt:.1f}, k={k_opt})', fontsize=12)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('p~')

# График 3: Контурные линии
ax3 = fig.add_subplot(133)
contour1 = ax3.contour(X1, X2, p_true_grid, levels=8, colors='blue', alpha=0.6, linewidths=1.5)
contour2 = ax3.contour(X1, X2, p_est_grid, levels=8, colors='red', alpha=0.6, linewidths=1.5, linestyles='dashed')
ax3.clabel(contour1, inline=True, fontsize=8)
ax3.set_title('Линии уровня\n(синий - истинная, красный - оценка)', fontsize=12)
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()