

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from model import model




def sample_senoidal(arg):
    return math.sin(2 * math.pi * arg) + random.uniform(-0.3 , 0.3) 


def generate_data(num_example_feature_xs):

    dataset = list()
    x_axis =  np.zeros((100), np.float)
    y_axis =  np.zeros((100), np.float)

    for i in range(0, num_example_feature_xs):

        x_coord_value = (i - 0)/(99 - 0)
        x_axis[i] = x_coord_value
        y_axis[i] = sample_senoidal(x_coord_value)

    dataset.append(x_axis)
    dataset.append(y_axis)
    return dataset


def plot_error_curve(error_registry):
    plt.plot(error_registry, linewidth = 3)
    plt.show()


def plot_model_function(model, num_examples):

    model_samples_y = np.zeros((num_examples), np.float)
    model_samples_x = np.zeros((num_examples), np.float)
    model_samples = list()

    for i in range(0, num_example_feature_xs):
        x_coord_value = (i - 0)/(99 - 0)
        model_samples_y[i] =  model.sample(x_coord_value)
        model_samples_x[i] = x_coord_value
    
    model_samples.append(model_samples_x)
    model_samples.append(model_samples_y)
    plot_dataset(model_samples)


def plot_dataset(dataset):
    plt.plot(dataset[0], dataset[1] ,linewidth = 2.0)
    plt.show()
        

def compute_regression_hypothesis(order, weights, feature_vector):

    dependent_variable = 0
    feature = feature_vector[0]
    feature_vector_index = 0
    power_index = 1

    for i in range(0, len(weights)):
        
        if power_index > 1:
            dependent_variable += weights[i] * pow(feature, power_index)
        else:
            dependent_variable += weights[i]

        if power_index == order:

            feature_vector_index += 1
            feature = feature_vector[feature_vector_index]
            power_index = 1    

    return dependent_variable


def run_gradient_descent(dataset, learning_rate = 0.0001, hypothesis_order = 3, num_features = 1):
    
    weights = np.random.uniform(low = -0.5, high = 0.5, size = (hypothesis_order * num_features))
    x_axis = dataset[0]
    y_axis = dataset[1]
    epsilon = 0.00001
    error_registry = list()
    print('\n\nPreparing Stochastic Gradient descent...')
    print("\nAlpha = ",learning_rate)
    print("\nEpsilon = ",epsilon)
    print('\n-------------------------------------------------------------')
    print('\n\n----------------Initializing Hell Descent...---------------')
    print('\n\n-----------------------------------------------------------')
    input("\n\nPress Enter to continue...")
    i = 0
    old_error = 0
    change_rate = 100

    while change_rate > epsilon:

        i += 1
        feature_vector = list()
        rand_index = int(random.uniform(0, len(x_axis) - 1))
        print(rand_index)
        example_feature_x = x_axis[rand_index]
        feature_vector.append(example_feature_x)
        error = sample_senoidal(example_feature_x) - compute_regression_hypothesis(hypothesis_order, weights, feature_vector)

        for weight in weights:
            weight = weight + learning_rate * (error * example_feature_x)
        
        print('\nStep: ',i)
        print('\n\nCost = ', error)
        error_registry.append(error)
        change_rate = error - old_error
        old_error = error
        
    print('\n\n\nFunction is now "epsilon exhausted"')
    print('Optimization is over.')
    print('Done.')

    return error_registry


def main():

    data = generate_data(100)
    error_registry = run_gradient_descent(data)
    plot_error_curve(error_registry)


if __name__ == "__main__":
    main()