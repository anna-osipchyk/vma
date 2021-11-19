import numpy as np
import random

np.set_printoptions(precision=20)


def generate(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                matrix[i][j] = random.randint(-100, 100)
                matrix[j][i] = matrix[i][j]
    for i in range(n):
        sum_a_i_j = sum(abs(matrix[i]))
        matrix[i][i] = random.randint(sum_a_i_j + 2, sum_a_i_j + 20)

    x = np.array(range(1, n + 1))
    f = np.dot(matrix, x)
    return matrix, x, f


def gradient_descent(matrix, f, e, x_pr):
    x0 = np.copy(f)
    x = np.copy(f)
    it = 0
    norm_r = 1
    while norm_r > e:
        it += 1
        r = np.dot(matrix, x) - f
        t = np.dot(r, r) / np.dot(np.dot(matrix, r), r)
        x = x - t * r
        norm_r = np.linalg.norm(np.dot(matrix, x) - f)
    return x, norm_r, it, x0, np.linalg.norm(x - x_pr)


def relaxation(matrix, w, f, e, n, x_abs):
    f_copy = np.copy(f)
    it = 0
    norm_r = 1
    x = np.zeros((n))
    x_prev = np.copy(f)
    x_next = np.zeros((n))
    while norm_r > e:
        for i in range(n):
            sum_next = 0
            for j in range(i):
                sum_next += matrix[i][j] * x_next[j] / matrix[i][i]
            sum_prev = 0
            for j in range(i + 1, n):
                sum_prev += matrix[i][j] * x_prev[j] / matrix[i][i]
            x[i] = (1 - w) * x_prev[i] - w * sum_next - w * sum_prev + w*f[i] / matrix[i][i]
            x_next[i] = x[i]
        x_prev = x
        it += 1
        norm_r = np.linalg.norm(np.dot(matrix, x) - f_copy)
    norm_x = np.linalg.norm(x - x_abs)
    return x, norm_r, it, norm_x


dim = 10
e = 10e-7
k = 5000
w = 0.2
matrix, x, f = generate(dim)

x_e, norm_r, it, x0, norm_x = gradient_descent(matrix, f, e, x)
with open("stats.txt", "w", encoding='utf-8') as file:
    stats = f"Матрица: \n{matrix}\n" \
            f"Точное решение: {x}\n" \
            f"Приближенное решение: {x_e}\n" \
            f"Итерация: {it}\n" \
            f"r: {norm_r}\n" \
            f"Норма погрешности решения: {norm_x}\n"
    print(stats)
    print(stats, file=file)
    for w in [0.2,0.5,0.8,1, 1.3, 1.5,1.8]:
        x_e, norm_r, it, norm_x  = relaxation(matrix, w, f, e, dim, x)
        stats = f"w: {w}\n" \
                f"Итерация: {it}\n" \
                f"r: {norm_r}\n" \
                f"Норма погрешности решения: {norm_x}\n"
        print(stats, file=file)


