import math
import matplotlib.pyplot as plt

def show_linear_result(theta, data, expected_price, predicted_price):
    plt.ylabel('Price')
    plt.xlabel('Kilometers')
    plt.scatter(data, expected_price)
    plt.plot(data.values.tolist(), predicted_price, color="red")
    plt.show()

def plot(metric_list, num_iters):
    plt.xlabel('Epoch')
    plt.plot(metric_list)

def show(legend):
    plt.legend(legend)
    plt.show()

'''
Calculate the difference between prediction and reality
Measure the variance
'''
def mean_squarred_error(training_data, expected_price, theta, nb_features, nb_examples):
    somme = 0
    for example_index in range(0, nb_examples):
        single_example_price = 0
        for feature_index in range(0, nb_features):
            single_example_price += (theta[feature_index] * training_data[example_index][feature_index])
        somme = expected_price[example_index] - single_example_price
    mse = (1 / nb_examples) * somme ** 2
    return mse[0]

'''
Calculate the difference between prediction and reality
Measure the standard deviation
'''
def root_mean_squarred_error(training_data, expected_price, theta, nb_features, nb_examples):
    mse = mean_squarred_error(training_data, expected_price, theta, nb_features, nb_examples)
    return math.sqrt(mse)

'''
Calculate the difference between prediction and reality
Measure the average difference
'''
def mean_average_error(training_data, expected_price, theta, nb_features, nb_examples):
    somme = 0
    for example_index in range(0, nb_examples):
        single_example_price = 0
        for feature_index in range(0, nb_features):
            single_example_price += (theta[feature_index] * training_data[example_index][feature_index])
        somme = abs(expected_price[example_index] - single_example_price)
    mae = (1 / nb_examples) * somme
    return mae[0]
'''
Apply the same transformation that has been used on the training data
'''
def scale_theta(theta, size, _max):
    # Loop start at 1 because it is associated with the bias that as not been through data scaling
    for i in range(1, size):
        theta[i] = (theta[i] - 0) / (_max)
    return theta
