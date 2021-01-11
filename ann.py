from decimal import Decimal
import numpy as np
import random


class Network:
    input_node_count = 0
    hidden_node_count = 0
    output_node_count = 0
    hidden_weights = []
    output_weights = []
    confusion_matrix = []
    error = 0.0

    # init that randomly generates weights
    def __init__(self, input_num, hidden_num, output_num):
        self.input_node_count = input_num
        self.hidden_node_count = hidden_num
        self.output_node_count = output_num
        self.hidden_weights = np.random.uniform(-1.0, 1.0, (hidden_num, input_num + 1))
        self.output_weights = np.random.uniform(-1.0, 1.0, (output_num, hidden_num + 1))
        self.confusion_matrix = generate_confusion_matrix(self.output_node_count)

    # init that takes a list of weights and biases from somewhere (i.e from the evolutionary algorithm) and assigns them
    def __init__(self, input_num, hidden_num, output_num, weights_and_bias):
        self.input_node_count = input_num
        self.hidden_node_count = hidden_num
        self.output_node_count = output_num
        self.hidden_weights = np.random.uniform(0, 0, (hidden_num, input_num + 1))
        self.output_weights = np.random.uniform(0, 0, (output_num, hidden_num + 1))
        self.confusion_matrix = generate_confusion_matrix(self.output_node_count)
        weight_count = 0
        for x in range(0, input_num + 1):
            for y in range(0, hidden_num):
                self.hidden_weights[y][x] = weights_and_bias[weight_count]
                weight_count += 1
        for x in range(0, hidden_num + 1):
            for y in range(0, output_num):
                self.output_weights[y][x] = weights_and_bias[weight_count]
                weight_count += 1


# given a network and it's count of nodes, generates the weights for them
# TODO: modify to use the node counts from the network class
def generate_weights(network, upper, lower, input_node_count, hidden_node_count, output_node_count):
    for t in range(0, input_node_count):
        for i in range(0, hidden_node_count):
            network.hidden_weights[t][i] = Decimal(random.uniform(lower, upper))

    for t in range(0, hidden_node_count):
        for i in range(0, output_node_count):
            network.hidden_weights[t][i] = Decimal(random.uniform(lower, upper))


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


# modifies a perceptrons weights based on the error and learning rate
def train_perceptron(weights, inputs, learning_rate, error):
    for x in range(1, len(weights)):
        weights[x] = weights[x] + (learning_rate * inputs[x] * error)


def simple_neural_algorithm(network, inputs, desired):
    data_size = len(inputs)
    # initialise the output list and assign zeroes to it to start with
    hidden_node_outputs = []
    for x in range(0, network.hidden_node_count):
        hidden_node_outputs.append(0)
    output_node_outputs = []
    for x in range(0, network.output_node_count):
        output_node_outputs.append(0)

    for t in range(0, data_size):  # for each data entry
        for i in range(0, network.hidden_node_count):  # for each hidden node
            hidden_node_outputs[i] = 0
            for j in range(0, network.input_node_count):  # for each input node
                # multiplying each input node value by it's weight for hidden node i
                hidden_node_outputs[i] += (network.hidden_weights[i][j] * inputs[t][j])
            hidden_node_outputs[i] += network.hidden_weights[i][network.input_node_count]  # bias
            hidden_node_outputs[i] = sigmoid_logistic(hidden_node_outputs[i])  # run sigmoid on final value (???)
        for i in range(0, network.output_node_count):  # for each output node
            output_node_outputs[i] = 0
            for j in range(0, network.hidden_node_count):  # for each hidden node
                # multiplying each hidden node output by it's weight for output node i
                output_node_outputs[i] += (network.output_weights[i][j] * hidden_node_outputs[j])
            output_node_outputs[i] += network.output_weights[i][network.hidden_node_count]  # bias
            output_node_outputs[i] = sigmoid_logistic(output_node_outputs[i])  # run sigmoid on final value (???)
        # converts output node values into probabilities
        test_softmax = softmax(output_node_outputs)
        # creates array of desired values for using in cross entropy
        desired_arr = get_desired_array(desired[t], network.output_node_count)
        # sets error using cross entropy
        network.error += cross_entropy(desired_arr, test_softmax)
        # update confusion matrix
        update_confusion_matrix(network.confusion_matrix, test_softmax, desired_arr)


def get_desired_array(desired, output_count):
    desired_arr = []
    for o in range(0, output_count):
        if o == desired:
            desired_arr.append(1)
        else:
            desired_arr.append(0)
    return desired_arr

def generate_confusion_matrix(outputs):
    confusion_matrix = []
    for x in range(0, outputs):
        confusion_matrix.append([0] * outputs)
    return confusion_matrix

def update_confusion_matrix(matrix, softmax, desired_arr):
    guess = 0
    prob = softmax[0]
    for x in range(1, len(softmax)):
        if softmax[x] > prob:
            guess = x
            prob = softmax[x]
    desired = desired_arr.index(max(desired_arr))
    matrix[desired][guess] += 1


# function I designed to calculate error by simply taking the difference between desired and actual output
def calculate_error_simple(network, desired, output):
    if desired == 1 and output < 0.5:
        network.error += 1.0
    if desired == 0 and output >= 0.5:
        network.error += 1.0


# adds the square error to network.error
def calculate_error_square_diff(network, desired, output):
    network.error += pow(desired - output, 2)


def calculate_mean_square_error(network, input_count):
    # network.error is already the sum of squared differences between desired and actual
    network.error = (1/input_count) * network.error


# softmax takes outputs from output nodes and converts them into probabilities
def softmax(outputs):
    return np.exp(outputs) / sum(np.exp(outputs))


# takes the target value and the actual value and calculates the cross entropy to be used as the error
def cross_entropy(target, actual):
    total = 0
    for e in range(0, len(target)):
        total += target[e] * np.log2(actual[e])
    return -total

def calculate_accuracy(confusion_matrix, total_tests):
    correct = 0
    incorrect = 0
    for x in range(0, len(confusion_matrix)):
        correct += confusion_matrix[x][x]
    incorrect = total_tests - correct
    accuracy = (100 / total_tests) * correct
    return accuracy
    print('eof')