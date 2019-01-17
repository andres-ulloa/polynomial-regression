

import numpy as np
import matplotlib.pyplot as plt
import random
import math


def sample_senoidal(arg):
    return math.sin(2 * math.pi * arg) + random.uniform(- 0.1 , 0.1) 


def generate_training_example():
    x_coord = random.uniform(0 ,1) 
    y_coord = sample_senoidal(x)
    return  np.array([x_coord , y_coord])


def generate_data(num_training_examples):
    dataset = np.zeros((100), np.float)
    for i in range(0, num_training_examples):
        training_example = generate_training_example()
        dataset[i] = training_example

    save_training_examples(dataset)
    return dataset


def save_training_examples(dataset):
   file = open('dataset.csv','w') 
   for example in dataset:
        file.write(example)
   file.close()


def plot_hypothesis():
    pass


def plot_dataset(dataset):
   pass


def compute_polynomial_linear_model(order, weights):
    for i in range(0, order):
        pass

def get_training_data(file_path):
    data = np.loadtxt('data.csv')
    return data

def train_linear_model():
    pass

def compute_cost_function():
    pass

def compute_gradient():
    pass

def main():
    new_data = input('¿Generate a new data set? (Y/N)')
    data =  np.zeros((100), np.float)
    if new_data == 'Y':
        data = generate_data()
    else:
        data = get_training_data('data.csv')
    

if __name__ == "__main__":
    main()