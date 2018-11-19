import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
data = pd.read_csv("ex1data2.csv")
Y = data["price"]
line, col = np.shape(data)

_min = np.reshape(np.array([[0.0], [0.0], [0.0]]), (col, 1))
_max = np.reshape(np.array([[0.0], [0.0], [0.0]]), (col, 1))
_mean = np.reshape(np.array([[0.0], [0.0], [0.0]]), (col, 1))

q = 0
for key in data:
    _min[q] = data[key].min()
    _max[q] = data[key].max()
    _mean[q] = data[key].mean()
    _lst = []
    for i in data[key]:
        _lst.append((i) / (_max[q] - _min[q]))
    q += 1
    data[key] = _lst

Y = np.reshape(Y, (np.size(Y), 1))
X = [np.insert(row, 0, 1) for row in data.drop(["price"], axis=1).values]
X = np.reshape(X, (line, col))

def somme(X, Y, theta, c, size, i):
    a = 0
    for j in range(0, size):
        a += (theta[j] * X[i][j])
    a -= Y[i]
    a = a * X[i][c]
    return (a)

def cost(X, theta, Y, c, alpha, size):
    a = 0
    for i in range(0, line):
        a += somme(X, Y, theta, c, size, i)
    return (theta[c] - ((alpha) * a))

def linear_reg(X, theta, Y, alpha, num_iters, size):
    temp = np.reshape(theta, (col, 1))
    for i in range(0, num_iters):
        if (i % 100 == 0):
            print(i)
        for j in range(0, size):
            temp[j] = cost(X, theta, Y, j, alpha, size)
        for j in range(0, size):
            theta[j] = temp[j]
    return (theta)

def scale_theta(theta, _min, _max, size):
    for i in range(1, size):
        theta[i] = (theta[i]) / (_max[i - 1] - _min[i - 1])
    return (theta)

def main():
    theta = [[0.0] * col]
    theta = np.reshape(theta, (col, 1))
    size = np.size(theta)
    alpha = 0.01
    num_iters = 1500
    theta = linear_reg(X, theta, Y, alpha, num_iters, size)
    print("\n", theta, "\n")
    theta = scale_theta(theta, _min, _max, size)
    print(theta[0], "\n", theta[1], " price\n", theta[2], " nb_bedrooms\n")

# TEST
    size_meters = 852
    nb_rooms = 2
    result = theta[0] + theta[1] * size_meters + theta[2] * nb_rooms
    print(result)

if __name__ == "__main__":
    main()
