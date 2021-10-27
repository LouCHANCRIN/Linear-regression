import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)
data = pd.read_csv("data.csv")
Y = data["price"]
line, col = np.shape(data)

_min = np.reshape(np.array([[0.0]] * col), (col, 1))
_max = np.reshape(np.array([[0.0]] * col), (col, 1))
_mean = np.reshape(np.array([[0.0]] * col), (col, 1))

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

X = [np.insert(row, 0, 1) for row in data.drop(["price"], axis=1).values]
X = np.reshape(X, (line, col))

def hypothese(X, Y, theta, index, size, i):
    a = 0
    # Calcul notre estimation pour un élément du dataset
    for j in range(0, size):
        a += (theta[j] * X[i][j])
    # Fait la différence entre notre résultat et la réalité
    a -= Y[i]
    a = a * X[i][index]
    return (a)

def somme(X, theta, Y, index, alpha, size):
    a = 0
    # Fait la somme des résultats de notre hypothèse sur l'ensemble du dataset
    for i in range(0, line):
        a += hypothese(X, Y, theta, index, size, i)
    return (theta[index] - ((alpha) * a))

def linear_reg(X, theta, Y, alpha, num_iters, size):
    temp = np.reshape(theta, (col, 1))
    for i in range(0, num_iters):
        # Calcul la nouvelle valeur de theta
        for index in range(0, size):
            temp[index] = somme(X, theta, Y, index, alpha, size)
        # Met à jour les theta après les avoir tous re calculé pour ne pas fausser les calculs
        for index in range(0, size):
            theta[index] = temp[index]
    return theta

def scale_theta(theta, _min, _max, size):
    for i in range(1, size):
        theta[i] = (theta[i]) / (_max[i - 1] - _min[i - 1])
    return theta

def main():
    theta = [[0.0] * col]
    theta = np.reshape(theta, (col, 1))
    size = np.size(theta)
    # ratio d'apprentissage
    alpha = 0.01
    num_iters = 1500
    theta = linear_reg(X, theta, Y, alpha, num_iters, size)
    theta = scale_theta(theta, _min, _max, size)

    with open('theta.txt', 'w') as f:
        f.write(f"{float(theta[0])}, {float(theta[1])}")


if __name__ == "__main__":
    main()
