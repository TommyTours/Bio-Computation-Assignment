from decimal import Decimal
import random
import ann

threshold = Decimal(random.uniform(-1.0, 1.0))  # originally tested with 0 as per coppin book

beginning_weights = [-threshold, Decimal(random.uniform(-1.0, 1.0)), Decimal(random.uniform(-1.0, 1.0))]

or_pairs = [[Decimal('0'), Decimal('0'), 0],
            [Decimal('0'), Decimal('1'), 1],
            [Decimal('1'), Decimal('0'), 1],
            [Decimal('1'), Decimal('1'), 1]]

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
