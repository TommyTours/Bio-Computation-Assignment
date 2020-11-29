from decimal import Decimal
import numpy as np


class network:
    hidden_weights = [], []
    output_weights = [], []
    error = 0.0

    def __init__(self, input_num, hidden_num, output_num):
        self.hidden_weights = [hidden_num][input_num + 1]
        self.output_weights = [output_num][hidden_num + 1]


def step_activation_function(inputs, weights, threshold):
    result = 0
    for x in range(0, len(inputs)):
        result += weights[x] * inputs[x]

    if result > threshold:
        return 1
    else:
        return 0


def sigmoid_logistic(x):
    return 1.0/(1 + np.exp(-x))


def logistic_deriv(x):
    return sigmoid_logistic(x) * (1 - sigmoid_logistic(x))


def train_perceptron(weights, inputs, learning_rate, error):
    for x in range(1, len(weights)):
        weights[x] = weights[x] + (learning_rate * inputs[x] * error)



