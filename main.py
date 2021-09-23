from random import randint
import numpy as np
from copy import deepcopy


def gen(dim: int):
    return list(range(1, dim + 1))


def generation(dim):
    matrix = [[randint(-100, 100) for _ in range(dim)] for _ in range(dim)]
    f = list()
    x = gen(dim)
    for equation in matrix:  # генерация вектора свободных членов
        el = sum(a * b for a, b in zip(equation, x))
        f.append(el)
    return matrix, f


def find_max_element(l_1: list, f: list, idx: int):
    list_of_column = [column[idx] for column in l_1]  # составление столбца
    max_el = max(list_of_column, key=abs)  # поиск ведущего элемента столбца
    index = list_of_column.index(max_el)
    l_1[index], l_1[idx] = l_1[idx], l_1[index]
    f[index], f[idx] = f[idx], f[index]  # замена строчек местами


def gauss(matrix: list, f: list):

    idx_of_main_element = 0

    for i in range(len(matrix) - 1):

        find_max_element(matrix, f, idx_of_main_element)
        el = matrix[i][idx_of_main_element]  # ведущий элемент
        matrix[i] = list(map(lambda x: x / el, matrix[i]))  # деление строчки на ведущий элемент
        f[i] /= el

        for j in range(i + 1, len(matrix)):
            first_el = matrix[j][idx_of_main_element]
            matrix[j] = list(map(lambda x1, x2: x2 - x1 * first_el, matrix[i], matrix[j]))
            # зануление всего столбца!!!
            f[j] -= f[i] * first_el

        idx_of_main_element += 1


    matrix[-1][-1], f[-1] = 1, f[-1] / matrix[-1][-1]

    answer = [0]
    counter = True

    for row, y in zip(reversed(matrix), reversed(f)):  # нахождение вектора неизвестных
        x = y - sum(a * b for a, b in zip(reversed(row), answer))
        if counter:
            answer.pop(0)
        counter = False
        answer.append(x)
    answer.reverse()
    return answer


def inaccuracy(x: list, answer: list):
    answ = max([a - b for a, b in zip(x, answer)]) / max(x)
    return answ


def steps(matrix, e_matrix, idx_of_main_element=0):

    for i in range(len(matrix) - 1):

        find_max_element(matrix, e_matrix, idx_of_main_element)
        el = matrix[i][idx_of_main_element]  # ведущий элемент
        matrix[i] = list(map(lambda x: x / el, matrix[i]))  # деление строчки на ведущий элемент
        e_matrix[i] = list(map(lambda x: x / el, e_matrix[i]))
        # у меня делится вся строчка!!!

        for j in range(i + 1, len(matrix)):
            first_el = matrix[j][idx_of_main_element]
            matrix[j] = list(map(lambda x1, x2: x2 - x1 * first_el, matrix[i], matrix[j]))
            # зануление всего столбца!!!
            e_matrix[j] = list(map(lambda x1, x2: x2 - x1 * first_el, e_matrix[i], e_matrix[j]))
            # зануление всего столбца!!!

        idx_of_main_element += 1


def inverse_matrix(matrix: list, dim: int):

    e_matrix = [[1 if i == j else 0 for i in range(dim)] for j in range(dim)]
    copy_of_matrix = deepcopy(matrix)

    steps(matrix, e_matrix)

    matrix[-1][-1], e_matrix[-1] = 1, list(map(lambda x: x / matrix[-1][-1], e_matrix[-1]))

    idx_of_main_element = len(matrix) - 1

    for i in range(len(matrix) - 1, 0, -1):

        el = matrix[i][idx_of_main_element]  # ведущий элемент
        matrix[i] = list(map(lambda x: x / el, matrix[i]))  # деление строчки на ведущий элемент
        e_matrix[i] = list(map(lambda x: x / el, e_matrix[i]))
        # у меня делится вся строчка!!!

        for j in range(i - 1, -1, -1):
            first_el = matrix[j][idx_of_main_element]
            matrix[j] = list(map(lambda x1, x2: x2 - x1 * first_el, matrix[i], matrix[j]))
            # зануление всего столбца!!!
            e_matrix[j] = list(map(lambda x1, x2: x2 - x1 * first_el, e_matrix[i], e_matrix[j]))
            # зануление всего столбца!!!

        idx_of_main_element -= 1

    A = np.array(copy_of_matrix)
    B = np.array(e_matrix)
    C = A.dot(B)
    return C, B


if __name__ == '__main__':
    dim = 10
    matrix, f = generation(dim)
    print('\nПорядок матрицы:', dim)
    print('\nСгенерированная матрица:')

    for el in matrix:
        print(el)

    e, inverse = inverse_matrix(matrix[:], dim)
    answ = gauss(matrix, f)
    print('\nПервоначальный столбец неизвестных:', gen(dim))
    print('\nПодсчитанный столбец неизвестных:', answ)
    print("\nОтносительная погрешность: ", inaccuracy(gen(dim), answ))
    print("\nОбратная матрица:\n", inverse)
    print("\nРезультат перемножения матрицы и обратной к ней:\n", e)
