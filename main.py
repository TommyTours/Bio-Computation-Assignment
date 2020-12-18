from decimal import Decimal
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import random
import ann
import evolution
import matplotlib.pyplot as plt


training_data = []
target_output = []
training_count = 0
x_train = []
y_train = []
x_test = []
y_test = []

def get_data_abalone():
    global training_data, target_output, training_count, x_train, y_train, x_test, y_test
    training_data = pandas.read_excel('abalone_data.xlsx')
    target_output = training_data.output
    training_data = training_data.drop(['output'], 1)
    training_data = np.asarray(training_data)
    training_count = len(training_data)

    for i in range(0, training_count):
        if target_output[i] == 'M':
            target_output[i] = 0
        elif target_output[i] == 'F':
            target_output[i] = 1
        else:
            target_output[i] = 2

    x = training_data[0:6, :]
    y = training_data[6, :]

    x_train, x_test, y_train, y_test = train_test_split(training_data, target_output, shuffle=False)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

def get_data_BB():
    global training_data, target_output, training_count, x_train, y_train, x_test, y_test
    training_data = pandas.read_excel('full_data.xlsx')
    target_output = training_data.output
    training_data = training_data.drop(['output'], 1)
    training_data = np.asarray(training_data)
    training_count = len(training_data[:, 0])

    x = training_data[0:6, :]
    y = training_data[6, :]

    x_train, x_test, y_train, y_test = train_test_split(training_data, target_output, shuffle=False)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)


def new_generation(population, mute_rate, mute_step, input_count, hidden_count, output_count, data, desired):

    offspring = []

    best_individual = evolution.get_best_nn(population)

    population = evolution.crossover(population)

    #evolution.individual_fitness_nn(population, input_count, hidden_count, output_count, data, desired)

    #population_fitness = evolution.population_fitness_nn(population)

    population = evolution.mutation(population, mute_rate, mute_step)

    evolution.individual_fitness_nn(population, input_count, hidden_count, output_count, data, desired)

    #population_fitness = evolution.population_fitness_nn(population)

    offspring = evolution.tournament_selection_nn(population)
    #offspring = evolution.variable_size_tournament(population, 5)

    evolution.replace_worst_with_best_nn(offspring, best_individual)

    #print("Pop average fitness: " + str(population_fitness[2]))

    return offspring


def evolution_test():
    population_size = 50
    upper = 1.0
    lower = -1.0
    #input_nodes = 10
    input_nodes = 8
    hidden_nodes = 4
    #output_nodes = 1
    output_nodes = 3
    mute_rate = 0.02
    mute_step = 1
    data_set = 'abalone'

    get_data_abalone()
    population = evolution.init_population_nn_weights(population_size, upper, lower, input_nodes, hidden_nodes, output_nodes)
    evolution.individual_fitness_nn(population, input_nodes, hidden_nodes, output_nodes, x_train, y_train)
    initial_fitness = evolution.population_fitness_nn(population, len(x_train))
    print("Initial average fitness: " + str(initial_fitness[2]))
    population = evolution.tournament_selection_nn(population)
    #population = evolution.variable_size_tournament(population, 5)
    tournament_fitness = evolution.population_fitness_nn(population, len(x_train))
    print("Fitness after first tournament: " + str(tournament_fitness[2]))

    best_and_mean = [[], []]
    test_set_error = []

    for x in range(0, 200):
        population = new_generation(population, mute_rate, mute_step, input_nodes, hidden_nodes, output_nodes, x_train, y_train)
        fitness = evolution.population_fitness_nn(population, len(x_train))
        print("average fitness after " + str(x+1) + " generations: " + str(fitness[2]))
        best_and_mean[0].append(fitness[1])
        best_and_mean[1].append(fitness[2])
        # test test data using best individual
        best_individual = evolution.get_best_nn(population)
        network = ann.Network(input_nodes, hidden_nodes, output_nodes, best_individual.gene)
        ann.simple_neural_algorithm(network, x_test, y_test)
        test_set_error.append(network.error / len(x_test))

    plt.title('Data set: ' + data_set + '\n' + 'Mute Rate: ' + str(mute_rate) + ', Mute Step: ' + str(mute_step) + '\npopulation size: ' + str(population_size) + ', hidden nodes: ' + str(hidden_nodes))
    plt.plot(best_and_mean[0])
    plt.plot(best_and_mean[1])
    plt.plot(test_set_error)
    plt.xlabel('Generation')
    plt.ylabel('Error')
    plt.legend(['Best', 'Mean', 'Test Error'])
    #plt.ylim(0.0, 1.0)
    plt.show()
    print("Best Training Fitness: " + str(best_and_mean[0][-1]))
    print("Final Test Set Error: " + str(test_set_error[-1]))

    print("eoc")

evolution_test()

