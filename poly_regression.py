

import numpy as np
import matplotlib.pyplot as plt
import random
import math


def sample_senoidal(arg):
    return math.sin(2 * math.pi * arg) + random.uniform(-0.3 , 0.3) 


def generate_data(num_training_examples):
    dataset = list()
    x_axis =  np.zeros((100), np.float)
    y_axis =  np.zeros((100), np.float)
    for i in range(0, num_training_examples):
        x_coord_value = (i - 0.01)/(0.99 - 0.01)
        x_axis[i] = x_coord_value
        y_axis[i] = sample_senoidal(x_coord_value)
    dataset.append(x_axis)
    dataset.append(y_axis)
    return dataset


def plot_dataset(dataset):
    plt.plot(dataset[0], dataset[1] ,linewidth = 2.0)
    plt.show()
        

def compute_hypothesis(order, weights, feature_vector):
    dependent_variable = 0
    for i in range(0, order * 3):
        pass
    return dependent_variable


def run_gradient_descent(dataset, learning_rate = 0.0001, hypothesis_order = 3):
    weights = np.random.uniform(low = -0.5, high = 0.5, size = hypothesis_order)
    x_axis = dataset[0]
    y_axis = dataset[1]
    features = y_axis
    for i in range(len(y_axis)):
        for weight in weights:
            x_coord = x_axis[i] 
            weight = weight + learning_rate * (sample_senoidal(x_coord) - compute_hypothesis(hypothesis_order, weights, feature_vector))


def main():
    data = generate_data(100)
    plot_dataset(data)
    train_linear_model(data)

if __name__ == "__main__":
    main()