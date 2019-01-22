

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


def plot_model_function(model, num_examples, dataset):

    model_samples_y = np.zeros((num_examples), np.float)
    model_samples_x = np.zeros((num_examples), np.float)
    model_samples = list()

    for i in range(0, num_examples):

        feature_vector = list()
        x_coord_value = (i - 0)/(99 - 0)
        feature_vector.append(x_coord_value)
        model_samples_y[i] =  model.sample_model(feature_vector)
        model_samples_x[i] = x_coord_value
    
    plt.plot(model_samples_x, model_samples_y ,linewidth = 2.0, color = 'C2')
    plt.scatter(dataset[0], dataset[1])
    plt.show()


def plot_dataset(dataset):
    plt.plot(dataset[0], dataset[1] ,linewidth = 2.0, color = 'C2')
    plt.scatter(dataset[0], dataset[1])
    plt.show()
        

def compute_hypothesis(order, weights, feature_vector):

    dependent_variable = 0
    feature_vector_index = 0
    power_index = 0

    for i in range(0, len(weights)):
          
        if i > 0:
            power_index += 1 
            dependent_variable += weights[i] * pow(feature_vector[feature_vector_index], power_index)
        else:
            dependent_variable += weights[i]

        if power_index == order and i + 1 < len(weights):
            
            feature_vector_index += 1
            feature = feature_vector[feature_vector_index]
            power_index = 0  
         

    return dependent_variable


def run_gradient_descent(dataset, learning_rate = 0.005, hypothesis_order = 3, num_features = 1, num_epochs = 5000000):
    
    weights = np.random.uniform(low = -0.5, high = 0.5, size = (hypothesis_order * num_features + num_features))
    print('Weights = ', weights)
    x_axis = dataset[0]
    y_axis = dataset[1]
    epsilon = 0.001
    error_registry = list()
    print('\n\nPreparing Stochastic Gradient descent...')
    print("\nAlpha = ",learning_rate)
    input("\n\nPress Enter to continue...")
    i = 0
    old_error = 0
    change_rate = 100

    for j in range(0, num_epochs):

        i += 1
        feature_vector = list()
        rand_index = int(random.uniform(0, len(x_axis) - 1))
        example_feature_x = x_axis[rand_index]
        example_feature_y = y_axis[rand_index]
        feature_vector.append(example_feature_x)

        error =  (example_feature_y - compute_hypothesis(hypothesis_order, weights, feature_vector)) 

        feature_vector_index = 0
        power_index = 0

        for x in range(0, len(weights)):
            feature_gradient = error * feature_vector[feature_vector_index]
            weights[x] = weights[x] + (learning_rate * feature_gradient)
            if power_index  == hypothesis_order and i + 1 < len(weights):
                power_index = 0
                feature_vector_index += 1 

            power_index += 1
        
        #print(weights)
        
        """print('\nStep: ',i)
        print('\n\nCost = ', error)"""
        error_registry.append(error)
        change_rate = abs(error - old_error)
        old_error = error
      
    
    regression_model = model(weights, hypothesis_order, error_registry)
    print('\n\n\nFunction is now "epsilon exhausted"')
    print('Optimization is over.')
    print('Done.')
    print(weights)
    return regression_model


def main():
    
    
    data = generate_data(100)
    poly_model = run_gradient_descent(data)
    plot_error_curve(poly_model.error_registry)
    plot_dataset(data)
    plot_model_function(poly_model, 100,data)
    
    """
    weights = [2,2,2,2]
    features = [3]
    print(compute_hypothesis(3, weights, features))
    """

if __name__ == "__main__":
    main()