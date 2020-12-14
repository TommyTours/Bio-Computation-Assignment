from decimal import Decimal
import numpy as np
import random


class Network:
    input_node_count = 0
    hidden_node_count = 0
    output_node_count = 0
    hidden_weights = []
    output_weights = []
    error = 0.0

    def __init__(self, input_num, hidden_num, output_num):  # init that randomly generates weights
        self.input_node_count = input_num
        self.hidden_node_count = hidden_num
        self.output_node_count = output_num
        self.hidden_weights = np.random.uniform(-1.0, 1.0, (hidden_num, input_num + 1))
        self.output_weights = np.random.uniform(-1.0, 1.0, (output_num, hidden_num + 1))

    def __init__(self, input_num, hidden_num, output_num, weights_and_bias):
        self.input_node_count = input_num
        self.hidden_node_count = hidden_num
        self.output_node_count = output_num
        self.hidden_weights = np.random.uniform(0, 0, (hidden_num, input_num + 1))
        self.output_weights = np.random.uniform(0, 0, (output_num, hidden_num + 1))
        weight_count = 0
        for x in range(0, input_num + 1):
            for y in range(0, hidden_num):
                self.hidden_weights[y][x] = weights_and_bias[weight_count]
                weight_count += 1
        for x in range(0, hidden_num + 1):
            for y in range(0, output_num):
                self.output_weights[y][x] = weights_and_bias[weight_count]
                weight_count += 1


def generate_weights(network, upper, lower, input_node_count, hidden_node_count, output_node_count):
    for t in range(0, input_node_count):
        network.hidden_weights
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


def train_perceptron(weights, inputs, learning_rate, error):
    for x in range(1, len(weights)):
        weights[x] = weights[x] + (learning_rate * inputs[x] * error)


def simple_neural_algorithm(network, inputs, desired):
    data_size = len(inputs)
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
                hidden_node_outputs[i] += (network.hidden_weights[i][j] * inputs[t][j])
            hidden_node_outputs[i] += network.hidden_weights[i][network.input_node_count]  # bias
            hidden_node_outputs[i] = sigmoid_logistic(hidden_node_outputs[i])
        for i in range(0, network.output_node_count):  # calculate output value for each output layer node
            output_node_outputs[i] = 0
            for j in range(0, network.hidden_node_count):
                output_node_outputs[i] += (network.output_weights[i][j] * hidden_node_outputs[j])
            output_node_outputs[i] += network.output_weights[i][network.hidden_node_count]  # bias
            output_node_outputs[i] = sigmoid_logistic(output_node_outputs[i])
        #if desired[t] == 1 and output_node_outputs[0] < 0.5:
        #    network.error += 1.0
        #if desired[t] == 0 and output_node_outputs[0] >= 0.5:
        #    network.error += 1.0
        #calcuate_error_simple(network, desired[t], output_node_outputs[0])
        #calculate_error_diff(network, desired[t], output_node_outputs[0])
        #calculate_error_multi_output(network, desired[t], output_node_outputs)
        test_softmax = softmax(output_node_outputs)
        desired_arr = get_desired_array(desired[t], network.output_node_count)
        test_cross_entropy = cross_entropy(desired_arr, test_softmax)
        calculate_error_square_diff(network, desired[t], output_node_outputs[0])

    calculate_mean_square_error(network, data_size)
    #print(output_node_outputs)


def calculate_error_multi_output(network, desired, outputs):
    desired_arr = get_desired_array(desired, network.output_node_count)

    get_desired_array(desired)
    for o in range(0, network.output_node_count):
        calculate_error_diff(network, desired_arr[o], outputs[o])


def get_desired_array(desired, output_count):
    desired_arr = []
    for o in range(0, output_count):
        if o == desired:
            desired_arr.append(1)
        else:
            desired_arr.append(0)
    return desired_arr


def calcuate_error_simple(network, desired, output):
    if desired == 1 and output < 0.5:
        network.error += 1.0
    if desired == 0 and output >= 0.5:
        network.error += 1.0
        
        
def calculate_error_diff(network, desired, output):
    if desired == 1:
        network.error += desired - output
    if desired == 0:
        network.error += output


def calculate_error_square_diff(network, desired, output):
    network.error += pow(desired - output, 2)


def calculate_mean_square_error(network, input_count):
    #network.error is already the sum of squared differences between desired and actual
    network.error = (1/input_count) * network.error


def softmax(outputs):
    return np.exp(outputs) / sum(np.exp(outputs))

def cross_entropy(target, actual):
    total = 0
    for e in range(0, len(target)):
        total += target[e] * np.log(actual[e])
    return -total


debug_training_data = [[0.803662, 0.981136, 0.369132, 0.498354, 0.067417, 0.067417, 0],
                       [0.193649, 0.519878, 0.563662, 0.38504, 0.395856, 0.553702, 1]]

#debug_training_data = [[0, 0, 0],
#                       [1, 0, 1],
#                       [0, 1, 1],
#                       [1, 1, 1]]
desired_output = []

#data_size = len(debug_training_data)

#input_node_count = 6
hidden_node_count = 3
output_node_count = 1

for i in range(0, len(debug_training_data)):
    desired_output.append(debug_training_data[i][-1])
    debug_training_data[i].remove(debug_training_data[i][-1])

weight_upper = 1.0
weight_lower = -1.0

#my_network = Network(input_node_count, hidden_node_count, output_node_count)

# generate_weights(my_network, weight_upper, weight_lower, input_node_count, hidden_node_count, output_node_count)

#simple_neural_algorithm(data_size, my_network, debug_training_data, hidden_node_count, output_node_count, desired_output)

#print("total error = " + str((my_network.error/(data_size/input_node_count))))
