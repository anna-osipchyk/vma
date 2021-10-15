import numpy as np
from random import randint

dim = 10


def generation():
    k = 2
    a = np.array([-randint(-100, 100) for _ in range(dim)])
    b = np.array([-randint(-100, 100) for _ in range(dim)])
    c = np.array([randint(abs(a_el) + abs(b_el) + k, abs(a_el) + abs(b_el) + 2 * k) for a_el, b_el in zip(a, b)])
    c[0], c[dim - 1] = randint(abs(b[0]) + k, abs(b[0]) + 2 * k), randint(abs(a[dim - 1]) + k, abs(a[dim - 1]) + 2 * k)
    a[0], b[dim - 1] = False, False
    y, f = np.array(list(range(1, dim + 1))), np.array([0 for _ in range(dim)])
    for i in range(1, dim - 1):
        f[i] = a[i] * y[i - 1] + c[i] * y[i] + b[i] * y[i + 1]
    f[0], f[dim - 1] = c[0] * y[0] + b[0] * y[1], a[dim - 1] * y[dim - 2] + c[dim - 1] * y[dim - 1]
    print("a: ", a)
    print("b: ", b)
    print("c: ", c)
    print("y: ", y)
    print("f: ", f)
    return a, b, c, y, f


def straight_stroke(a, b, c, f):
    alpha, betta = np.zeros((1, dim + 1))[0], np.zeros((1, dim + 1))[0]
    alpha[1], betta[1] = -(b[0]) / c[0], f[0] / c[0]
    for i in range(1, dim):
        denominator = c[i] - (-1) * a[i] * alpha[i]
        alpha[i + 1], betta[i + 1] = -b[i] / denominator, (f[i] + (-1) * a[i] * betta[i]) / denominator

    # betta[dim] = (f[dim-1] + abs(a[dim-1])*betta[dim-1])/(c[dim-1] - abs(a[dim-1])*alpha[dim-1])
    return alpha, betta


def reverse_stroke(alpha, betta):
    # x = np.zeros((1,dim+1))[0]
    x = [0 for _ in range(dim + 1)]
    x[dim - 1] = betta[dim]

    for i in range(dim - 1, -1, - 1):
        x[i] = alpha[i + 1] * x[i + 1] + betta[i + 1]
    return x[:len(x) - 1]


a, b, c, y, f = generation()
alpha, betta = straight_stroke(a, b, c, f)
alpha, betta = np.asarray(alpha), np.asarray(betta)
x = reverse_stroke(alpha, betta)
eps = max(a - b for a, b in zip(x, y)) / max(y)
print("x: ", x)
print("Погрешность: ", eps)
