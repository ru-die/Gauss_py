import numpy as np


def gaussian_elimination(A, b):

    n = len(b)
    Ab = np.concatenate((A, b.reshape((n, 1))), axis=1)

    for k in range(n - 1):
        # поиск строки с максом
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        if Ab[max_row, k] == 0:
            return None

        # поменять строки местами
        Ab[[k, max_row]] = Ab[[max_row, k]]

        # Обнуление элементов ниже главной диагонали
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]


    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

slau = int(input("Введите количество уравнений: "))
col_s = slau

A = np.zeros((slau, col_s))
b = np.zeros(slau)

print("Введите коэффициенты матрицы A:")
for i in range(slau):
    A[i] = list(map(float, input().split()))

print("Введите вектор свободных членов b:")
b = np.array(list(map(float, input().split())))

# Решение слау
x = gaussian_elimination(A, b)

if x is not None:
    print("Решение:")
    for i in range(len(x)):
        print(f"x{i + 1} = {round(x[i])}")
else:
    print("Система уравнений не имеет единственного решения.")
