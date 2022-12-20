import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance_matrix(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = round(random.uniform(10, 99), 1)
    return matrix


def calculate_shortest_paths(matrix):
    path_list = []
    n = len(matrix)
    for i in range(n):
        shortest = float('inf')
        shortest_index = -1
        for j in range(i + 1, n):
            if (matrix[i][j]) < shortest:
                shortest = matrix[i][j]
                shortest_index = j
        for j in range(i + 1, n):
            if j != shortest_index:
                matrix[i][j] = matrix[j][i] = 0
            else:
                path_list.append((i, j, matrix[i][j]))
    return matrix, sorted(path_list, key=lambda x: x[-1], reverse=True)


def calculate_final_graph(matrix, paths):
    for path in paths:
        for i in range(n):
            for j in range(i + 1, n):
                if i == path[0] and j == path[1]:
                    matrix[i][j] = matrix[j][i] = 0
                    print(f'Removed ({path[0]}, {path[1]}) {path[2]}')
    return matrix


def draw_graph(G):
    plt.figure(1, figsize=(16, 9))
    pos = nx.spring_layout(G, k=0.7 / np.sqrt(len(G.nodes())))
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == '__main__':
    n = 10
    matrix = calculate_distance_matrix(n)
    G = nx.from_numpy_matrix(np.asmatrix(np.array(matrix)))
    draw_graph(G)

    shortest_matrix, paths = calculate_shortest_paths(matrix)
    G_shortest = nx.from_numpy_matrix(np.asmatrix(shortest_matrix))
    draw_graph(G_shortest)

    d = 0
    k = 0
    lengths = list(map(lambda x: x[-1], paths))
    for i in range(1, len(lengths) - 1):
        new_d = abs(lengths[i] - lengths[i - 1])
        if new_d > d:
            d = new_d
            k = i
    print(f'Clusters: {k + 1}')

    final_matrix = calculate_final_graph(shortest_matrix, paths[0:k])
    G_final = nx.from_numpy_matrix(np.asmatrix(final_matrix))
    draw_graph(G_final)