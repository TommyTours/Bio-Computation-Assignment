from decimal import Decimal
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import random
import ann
import evolution
import matplotlib.pyplot as plt


#pandas_test = pandas.read_excel('training_data.xlsx')
#output = pandas_test.output
#pandas_test = pandas_test.drop(['output'], 1)
#pandas_test = np.asarray(pandas_test)
#weightsItoH = np.random.uniform(-1, 1, (3, 4))
#weightsHtoO = np.random.uniform(-1, 1, 4)
#testval = pandas_test[0,:]
#preactH = np.zeros(4)
#postactH = np.zeros(4)

training_data = pandas.read_excel('full_data.xlsx')
target_output = training_data.output
training_data = training_data.drop(['output'], 1)
training_data = np.asarray(training_data)
training_count = len(training_data[:, 0])

x = training_data[0:6, :]
y = training_data[6, :]

x_train, x_test, y_train, y_test = train_test_split(training_data, target_output)

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

    #offspring = evolution.tournament_selection_nn(population)
    offspring = evolution.variable_size_tournament(population, 5)

    evolution.replace_worst_with_best_nn(offspring, best_individual)

    #print("Pop average fitness: " + str(population_fitness[2]))

    return offspring


def evolution_test():
    population_size = 50
    upper = 1.0
    lower = -1.0
    input_nodes = 10
    hidden_nodes = 3
    output_nodes = 1
    mute_rate = 0.02
    mute_step = 1

    population = evolution.init_population_nn_weights(population_size, upper, lower, input_nodes, hidden_nodes, output_nodes)
    evolution.individual_fitness_nn(population, input_nodes, hidden_nodes, output_nodes, x_train, y_train)
    initial_fitness = evolution.population_fitness_nn(population, len(x_train))
    print("Initial average fitness: " + str(initial_fitness[2]))
    #population = evolution.tournament_selection_nn(population)
    population = evolution.variable_size_tournament(population, 5)
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
        test_set_error.append(network.error/ len(x_test))

    plt.title('Mute Rate: ' + str(mute_rate) + ', Mute Step: ' + str(mute_step))
    plt.plot(best_and_mean[0])
    plt.plot(best_and_mean[1])
    plt.plot(test_set_error)
    plt.xlabel('Generation')
    plt.ylabel('Error')
    plt.legend(['Best', 'Mean','Test Error'])
    #plt.ylim(0.0, 1.0)
    plt.show()
    print("Best: " + str(best_and_mean[0][-1]))

    print("eoc")


def nn_test():
    threshold = Decimal(random.uniform(-1.0, 1.0))  # originally tested with 0 as per coppin book

    beginning_weights = [-threshold, Decimal(random.uniform(-1.0, 1.0)), Decimal(random.uniform(-1.0, 1.0))]

    learning_rate = 1

    input_nodes = 10
    hidden_nodes = 3
    output_nodes = 1

    epoch_count = 4

    my_network = ann.Network(input_nodes, hidden_nodes, output_nodes)

    weights_input_to_hidden = np.random.uniform(-1, 1, (input_nodes, hidden_nodes))
    weights_hidden_to_output = np.random.uniform(-1, 1, hidden_nodes)

    pre_activation_hidden = np.zeros(hidden_nodes)
    post_activation_hidden = np.zeros(hidden_nodes)


    ann.simple_neural_algorithm(my_network, x_train, y_train)

    #validation_data = pandas.read_excel('assignment_validation_data_1.xlsx')
    #validation_output = validation_data.output
    #validation_data = validation_data.drop(['output'], 1)
    #validation_data = np.asarray(validation_data)
    #validation_count = len(validation_data[:, 0])

    training_data = [[0, 0],
                     [1, 0],
                     [0, 1],
                     [1, 1]]

    training_data = np.asarray(training_data)

    target_output = [0,
                     1,
                     1,
                     1]

    training_count = len(training_data)
    print('eoc')


def mlp():
    for epoch in range(epoch_count):
        for sample in range(training_count):
            for hidden_node in range(hidden_nodes):
                pre_activation_hidden[hidden_node] = np.dot(training_data[sample, :], weights_input_to_hidden[:, hidden_node])
                post_activation_hidden[hidden_node] = ann.sigmoid_logistic(pre_activation_hidden[hidden_node])

            pre_activation_output = np.dot(post_activation_hidden, weights_hidden_to_output)
            post_activation_output = ann.sigmoid_logistic(pre_activation_output)

            final_error = post_activation_output - target_output[sample]

            for hidden_node in range(hidden_nodes):
                s_error = final_error * ann.logistic_deriv(pre_activation_output)
                gradient_hidden_to_output = s_error * post_activation_hidden[hidden_node]

                for input_node in range(input_nodes):
                    input_value = training_data[sample, input_node]
                    gradient_input_to_hidden = s_error * weights_hidden_to_output[hidden_node] * ann.logistic_deriv(pre_activation_hidden[hidden_node]) * input_value

                    weights_input_to_hidden[input_node, hidden_node] = learning_rate * gradient_input_to_hidden

                weights_hidden_to_output[hidden_node] -= learning_rate * gradient_hidden_to_output


    ##################
    # validation
    ##################
    correct_classification_count = 0
    for sample in range(validation_count):
        for node in range(hidden_nodes):
            pre_activation_hidden[node] = np.dot(validation_data[sample, :], weights_input_to_hidden[:, node])
            post_activation_hidden[node] = ann.sigmoid_logistic(pre_activation_hidden[node])

        pre_activation_output = np.dot(post_activation_hidden, weights_hidden_to_output)
        post_activation_output = ann.sigmoid_logistic(pre_activation_output)

        if post_activation_output > 0.5:
            output = 1
        else:
            output = 0

        if output == validation_output[sample]:
            correct_classification_count += 1

    print('Percentage of correct classifications:')
    print(correct_classification_count*100/validation_count)

def solver_or():
    or_pairs = [[Decimal('0'), Decimal('0'), 0],
                [Decimal('0'), Decimal('1'), 1],
                [Decimal('1'), Decimal('0'), 1],
                [Decimal('1'), Decimal('1'), 1]]

    test = or_pairs[:0]

    print("Threshold: " + str(threshold))
    learning_rate = Decimal('0.1')
    weights = beginning_weights  # initially tested with -0.2 and 0.4 as per coppin book

    solved = False

    print('or weights at start: ' + str(weights[1]) + ', ' + str(weights[2]))
    run = 1
    while not solved:
        any_errors = False
        for x in range(0, len(or_pairs)):
            inputs = [Decimal('1'), or_pairs[x][0], or_pairs[x][1]]
            actual = ann.step_activation_function(inputs, weights, threshold)
            if actual != or_pairs[x][2]:
                any_errors = True
                error = or_pairs[x][2] - actual
                ann.train_perceptron(weights, inputs, learning_rate, error)
        print('or weights after ' + str(run) + ' runs: ' + str(weights[1]) + ', ' + str(weights[2]))
        run += 1
        if not any_errors:
            solved = True

    print('0 or 0 = ' + str(ann.step_activation_function([Decimal('1'), or_pairs[0][0], or_pairs[0][1]], weights, threshold)))
    print('1 or 0 = ' + str(ann.step_activation_function([Decimal('1'), or_pairs[1][0], or_pairs[1][1]], weights, threshold)))
    print('0 or 1 = ' + str(ann.step_activation_function([Decimal('1'), or_pairs[2][0], or_pairs[2][1]], weights, threshold)))
    print('1 or 1 = ' + str(ann.step_activation_function([Decimal('1'), or_pairs[3][0], or_pairs[3][1]], weights, threshold)))

def solve_and():
    and_pairs = [[Decimal('0'), Decimal('0'), 0],
                 [Decimal('0'), Decimal('1'), 0],
                 [Decimal('1'), Decimal('0'), 0],
                 [Decimal('1'), Decimal('1'), 1]]

    threshold = Decimal('1')
    learning_rate = Decimal('0.2')
    weights = beginning_weights#[-threshold, Decimal('-0.2'), Decimal('0.4')]

    solved = False
    print('and weights at start: ' + str(weights[1]) + ', ' + str(weights[2]))
    run = 1
    while not solved:
        any_errors = False
        for x in range(0, len(or_pairs)):
            inputs = [Decimal('1'), and_pairs[x][0], and_pairs[x][1]]
            actual = ann.step_activation_function(inputs, weights, threshold)
            if actual != and_pairs[x][2]:
                any_errors = True
                error = and_pairs[x][2] - actual
                ann.train_perceptron(weights, inputs, learning_rate, error)
        run += 1
        print('and weights after ' + str(run) + ' runs: ' + str(weights[1]) + ', ' + str(weights[2]))
        if not any_errors:
            solved = True

    print('0 and 0 = ' + str(ann.step_activation_function([Decimal('1'), and_pairs[0][0], and_pairs[0][1]], weights, threshold)))
    print('1 and 0 = ' + str(ann.step_activation_function([Decimal('1'), and_pairs[1][0], and_pairs[1][1]], weights, threshold)))
    print('0 and 1 = ' + str(ann.step_activation_function([Decimal('1'), and_pairs[2][0], and_pairs[2][1]], weights, threshold)))
    print('1 and 1 = ' + str(ann.step_activation_function([Decimal('1'), and_pairs[3][0], and_pairs[3][1]], weights, threshold)))


evolution_test()

