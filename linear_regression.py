import sys
import utils
import argparse
import numpy as np
import pandas as pd

# ratio d'apprentissage
alpha = 0.01
num_iters = 10000

# Lecture de la donnée
data = pd.read_csv("data.csv")
# data = pd.read_csv("ex1data2.csv")

# Extrait les prix attendues
expected_price = data["price"]

nb_examples, col = np.shape(data)

df = data.drop('price', axis=1)

_max = df.max().values[0]
# Réduit la donnée entre 0 et 1 pour éviter les overflows
scaled_data = (df) / (_max)

# Ajoute un colonne remplie avec des 0 pour le biais
training_data = [np.insert(row, 0, 1) for row in scaled_data.values]
training_data = np.reshape(training_data, (nb_examples, col))


'''
Calcule la différence entre le résultat obtenu et celui attendu pour un exemple
unique de données
'''
def hypothese(theta, feature_index, nb_features, ex_index):
    estimation = 0
    # Calcul notre estimation pour un élément du dataset
    for j in range(0, nb_features):
        estimation += (theta[j] * training_data[ex_index][j])
    # Fait la différence entre notre résultat et la réalité
    difference = estimation - expected_price[ex_index]
    result = difference * training_data[ex_index][feature_index]
    return result

'''
Fait la somme de la différence entre le résultat obtenu et celui attendu pour
l'ensemble du jeux de données
'''
def somme(theta, feature_index, nb_features):
    somme = 0
    # Fait la somme des résultats de notre hypothèse sur l'ensemble du dataset
    for ex_index in range(0, nb_examples):
        somme += hypothese(theta, feature_index, nb_features, ex_index)
    return somme

'''
Calcul la nouvelle valeur des thetas et les mets à jour simultanément
'''
def linear_reg(theta, nb_features, args):
    temp = np.reshape(theta, (col, 1))
    if args.early_stopping:
        if args.mse:
            early_stopping_metric = utils.mean_squarred_error
            early_stopping_legend = 'MSE'
        elif args.rmse:
            early_stopping_metric = utils.root_mean_squarred_error
            early_stopping_legend = 'RMSE'
        elif args.mae:
            early_stopping_metric = utils.mean_average_error
            early_stopping_legend = 'MAE'
        metric_list = []
    else:
        if args.mse:
            mse = []
        if args.rmse:
            rmse = []
        if args.mae:
            mae = []

    for i in range(0, num_iters):
        # Calcul la nouvelle valeur de chaque theta
        for feature_index in range(0, nb_features):
            temp[feature_index] = theta[feature_index] - alpha * somme(theta, feature_index, nb_features)
        # Met à jour tout les thetas après les avoir tous calculé pour cette itération
        for feature_index in range(0, nb_features):
            theta[feature_index] = temp[feature_index]

        if args.early_stopping:
            current_metric_value = early_stopping_metric(training_data, expected_price, theta, nb_features, nb_examples) 
            if len(metric_list) > 0:
                # If current error metric value is higher that the previous one, we stop the training
                if current_metric_value > metric_list[-1]:
                    print(f"Early stopping stopped the training loop after {i} iterations. Current MSE is {current_metric_value} and the previous one is {metric_list[-1]}")
                    metric_list.append(current_metric_value)
                    utils.plot(metric_list, i)
                    utils.show([early_stopping_legend])
                    return theta
            metric_list.append(current_metric_value)
            if i == num_iters - 1:
                metric_list.append(current_metric_value)
                utils.plot(metric_list, i)
                utils.show([early_stopping_legend])
                return theta
        else:
            if args.mse:
                mse.append(utils.mean_squarred_error(training_data, expected_price, theta, nb_features, nb_examples))
            if args.rmse:
                rmse.append(utils.root_mean_squarred_error(training_data, expected_price, theta, nb_features, nb_examples))
            if args.mae:
                mae.append(utils.mean_average_error(training_data, expected_price, theta, nb_features, nb_examples))
        
    if [args.mse, args.rmse, args.mae].count(True) > 0:
        legend = []
        if args.mse:
            utils.plot(mse, i)
            legend.append('MSE')
        if args.rmse:
            utils.plot(rmse, i)
            legend.append('RMSE')
        if args.mae:
            utils.plot(mae, i)
            legend.append('MAE')
        
        utils.show(legend)

    return theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse for bonus')
    parser.add_argument('--early_stopping', dest='early_stopping', default=False, action='store_true', help='Stop training when no progess is being made')
    parser.add_argument('--mse', dest='mse', default=False, action='store_true', help='Use the mean squared error')
    parser.add_argument('--rmse', dest='rmse', default=False, action='store_true', help='Use the root mean squared error')
    parser.add_argument('--mae', dest='mae', default=False, action='store_true', help='Use the mean average error')
    parser.add_argument('--show_linear_result', dest='show_linear_result', default=False, action='store_true', help='Show the linear function and the training data')

    args = parser.parse_args()

    if args.early_stopping and [args.mse, args.rmse, args.mae].count(True) != 1:
        sys.exit("Early stopping must be used with one and only one metric")


    # Initialisation de théta à 0
    theta = [[0.0] * col]
    theta = np.reshape(theta, (col, 1))

    nb_features = np.size(theta)
    theta = linear_reg(theta, nb_features, args)

    # Applique la réduction sur les poids pour car on s'entraine avec
    # des input réduite pour attendre des prix non réduits donc on doit adapter les theta
    theta = utils.scale_theta(theta, nb_features, _max)

    if args.show_linear_result:
        predicted_price = [[float(theta[0] + theta[1] * x)] for x in df.values]
        utils.show_linear_result(theta, df, expected_price, predicted_price)

    with open('theta.txt', 'w') as f:
        theta_string = str(float(theta[0]))
        for i in range(1, len(theta)):
            theta_string += f', {float(theta[i])}'
        f.write(theta_string)
        # f.write(f"{float(theta[0])}, {float(theta[1])}")