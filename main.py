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

def get_data(data_name, shuffle_data):
    global training_data, target_output, training_count, x_train, y_train, x_test, y_test
    training_data = pandas.read_excel(data_name)
    target_output = training_data.output
    training_data = training_data.drop(['output'], 1)
    training_data = np.asarray(training_data)
    training_count = len(training_data)  # previously had this, idk why: [:, 0])

    x_train, x_test, y_train, y_test = train_test_split(training_data, target_output,  test_size=0.33, shuffle=shuffle_data)

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
    hidden_nodes = 4
    #output_nodes = 1
    #output_nodes = 3
    #output_nodes = 2
    mute_rate = 0.02
    mute_step = 0.1
    shuffle_data = False
    #data_set = 'abalone'
    data_set = 'BB_data2.xlsx'

    #get_data_abalone()
    #get_data_BB()
    #get_data_diabetes()
    get_data(data_set, shuffle_data)
    input_nodes = len(training_data[0])
    output_nodes = len(np.unique(target_output))
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
    test_set_accuracy = []

    for x in range(0, 400):
        population = new_generation(population, mute_rate, mute_step, input_nodes, hidden_nodes, output_nodes, x_train, y_train)
        fitness = evolution.population_fitness_nn(population, len(x_train))
        print("Total fitness after " + str(x+1) + " generations: " + str(fitness[0]))
        best_and_mean[0].append(fitness[1])
        best_and_mean[1].append(fitness[2])
        # test test data using best individual
        best_individual = evolution.get_best_nn(population)
        network = ann.Network(input_nodes, hidden_nodes, output_nodes, best_individual.gene)
        ann.simple_neural_algorithm(network, x_test, y_test)
        test_set_accuracy.append(ann.calculate_accuracy(network.confusion_matrix, len(x_test)))
        test_set_error.append(network.error / len(x_test))

    plt.figure(1)

    plt.title('Data set: ' + data_set + '\n' + 'Mute Rate: ' + str(mute_rate) + ', Mute Step: ' + str(mute_step) + '\npopulation size: ' + str(population_size) + ', hidden nodes: ' + str(hidden_nodes) + '\nData Shuffled: ' + str(shuffle_data))
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
    
    plt.figure(2)
    plt.title('Test Set Accuracy')
    plt.plot(test_set_accuracy)
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.show()

    print('Final test set confusion matrix:')
    for x in range(len(network.confusion_matrix)):
        print(network.confusion_matrix[x], end=",")
    print('')
    print('Final Test Set Accuracy: ' + str(test_set_accuracy[-1]) + '%')

    print("eoc")

evolution_test()

