import numpy as np


def calculate_weights(input_matrix):
    num_neurons = input_matrix.size
    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if i != j:
                weights[i, j] = (2 * input_matrix.flatten()[i] - 1) * (2 * input_matrix.flatten()[j] - 1)

    return weights


def run_hopfield_network(input_matrix, weights):
    input_matrix = input_matrix.reshape(-1, 1)
    output_matrix = np.dot(weights, input_matrix)
    output_matrix = np.where(output_matrix > 0, 1, output_matrix)
    output_matrix = np.where(output_matrix == 0, input_matrix, output_matrix)
    output_matrix = np.where(output_matrix < 0, 0, output_matrix)

    return output_matrix


def display_matrix(matrix):
    for row in matrix:
        row_str = ""
        for elem in row:
            if elem == 1:
                row_str += "X"
            else:
                row_str += " "
        print(row_str)



input_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 1, 1, 0, 0, 1, 0],
                       [0, 1, 0, 0, 1, 0, 1, 0],
                       [0, 1, 0, 0, 0, 0, 1, 0],
                       [0, 0, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

display_matrix(input_data)
input_data.flatten()

weights = calculate_weights(input_data)

test_data = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])

display_matrix(input_data)

output_data = run_hopfield_network(test_data, weights)

print("\nWynik wyjściowy:")
display_matrix(output_data.reshape(8, 8))