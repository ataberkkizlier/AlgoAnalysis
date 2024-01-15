import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Global variables for Ant Colony Optimization
best_path = None
best_path_length = float('inf')

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    adjacency_str, bandwidth_str, delay_str, reliability_str = '\n'.join(lines).split('\n\n')

    adjacency_matrix = np.array([list(map(int, row.split(':'))) for row in adjacency_str.strip().split('\n')])
    bandwidth_matrix = np.array([list(map(int, row.split(':'))) for row in bandwidth_str.strip().split('\n')])
    delay_matrix = np.array([list(map(int, row.split(':'))) for row in delay_str.strip().split('\n')])
    reliability_matrix = np.array([list(map(float, row.split(':'))) for row in reliability_str.strip().split('\n')])

    return adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix

def constraints_check(path, bandwidth_matrix, delay_matrix, reliability_matrix):
    if not path:
        return False, False, False

    path_edges = list(zip(path[:-1], path[1:]))
    bandwidth_values = [bandwidth_matrix[u][v] for u, v in path_edges]
    reliability_values = [reliability_matrix[u][v] for u, v in path_edges]

    min_bandwidth = np.min(bandwidth_values)
    total_delay = np.sum(delay_matrix[np.array(path)])
    min_reliability = np.min(reliability_values)

    print("Bandwidth values along the path:", bandwidth_values)

    bandwidth_constraint = min_bandwidth >= 5
    delay_constraint = total_delay < 40
    reliability_constraint = min_reliability > 0.70

    return bandwidth_constraint, delay_constraint, reliability_constraint

def dijkstra_algorithm(graph, source, destination):
    distance, path = nx.single_source_dijkstra(graph, source, target=destination)
    return path

def bellman_ford_algorithm(graph, source, destination):
    path = nx.bellman_ford_path(graph, source, target=destination)
    return path

def a_star_algorithm(graph, source, destination):
    path = nx.astar_path(graph, source, destination)
    return path

def ant_colony_algorithm(graph, source, destination, iterations=100, ants=10, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9):
    num_nodes = len(graph.nodes)
    pheromone_matrix = np.ones((num_nodes, num_nodes))
    best_path = None
    best_path_length = float('inf')

    for iteration in range(iterations):
        ant_paths = []

        for ant in range(ants):
            current_node = source
            path = [current_node]

            while current_node != destination:
                next_node = select_next_node(graph, current_node, pheromone_matrix, alpha, beta, q0)
                path.append(next_node)
                current_node = next_node

            ant_paths.append(path)

        update_pheromones(pheromone_matrix, ant_paths, graph, evaporation)
        update_best_path(ant_paths, graph, destination)

    return best_path

def select_next_node(graph, current_node, pheromone_matrix, alpha, beta, q0):
    neighbors = list(graph.neighbors(current_node))
    probabilities = []

    for neighbor in neighbors:
        pheromone = pheromone_matrix[current_node][neighbor]
        visibility = 1 / graph[current_node][neighbor]['weight']
        probabilities.append((pheromone**alpha) * (visibility**beta))

    if np.random.rand() < q0:
        max_prob_index = np.argmax(probabilities)
        return neighbors[max_prob_index]
    else:
        probabilities /= sum(probabilities)
        chosen_node = np.random.choice(neighbors, p=probabilities)
        return chosen_node

def update_pheromones(pheromone_matrix, ant_paths, graph, evaporation):
    pheromone_matrix *= (1 - evaporation)

    for path in ant_paths:
        path_length = calculate_path_length(path, graph)
        pheromone_to_deposit = 1 / path_length

        for u, v in zip(path[:-1], path[1:]):
            pheromone_matrix[u][v] += pheromone_to_deposit
            pheromone_matrix[v][u] += pheromone_to_deposit

def update_best_path(ant_paths, graph, destination):
    global best_path, best_path_length

    for path in ant_paths:
        path_length = calculate_path_length(path, graph)

        if path[-1] == destination and path_length < best_path_length:
            best_path = path
            best_path_length = path_length

def calculate_path_length(path, graph):
    return sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))

def create_graph(adjacency_matrix):
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])

    return G

def display_shortest_path(graph, shortest_path, bandwidth_matrix, delay_matrix, reliability_matrix):
    pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue')

    # Highlight the shortest path
    edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
    edge_labels = {(u, v): (graph[u][v]['weight'], bandwidth_matrix[u][v], delay_matrix[u][v], reliability_matrix[u][v]) for u, v in edges}
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=2, edge_color='red', arrowsize=20)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.show()

def Solution(file_path, source, destination, bandwidth_requirement):
    adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix = read_input_file(file_path)

    G = create_graph(adjacency_matrix)

    print("Graph nodes:", G.nodes)

    path_dijkstra = dijkstra_algorithm(G, source, destination)

    bandwidth_constraint, delay_constraint, reliability_constraint = constraints_check(
        path_dijkstra, bandwidth_matrix, delay_matrix, reliability_matrix
    )

    if not (bandwidth_constraint and delay_constraint and reliability_constraint):
        print("Dijkstra: The path doesn't satisfy your constraints:")
        if not bandwidth_constraint:
            print("- The bandwidth constraint is not met")
        if not delay_constraint:
            print("- The delay constraint is not met")
        if not reliability_constraint:
            print("- The reliability constraint is not met")
    else:
        print("Dijkstra: It works fine.")
        display_shortest_path(G, path_dijkstra, bandwidth_matrix, delay_matrix, reliability_matrix)

    path_bellman = bellman_ford_algorithm(G, source, destination)

    bandwidth_constraint, delay_constraint, reliability_constraint = constraints_check(
        path_bellman, bandwidth_matrix, delay_matrix, reliability_matrix
    )

    if not (bandwidth_constraint and delay_constraint and reliability_constraint):
        print("Bellman-Ford: Path doesn't satisfy constraints:")
        if not bandwidth_constraint:
            print("- Bandwidth constraint not met")
        if not delay_constraint:
            print("- Delay constraint not met")
        if not reliability_constraint:
            print("- Reliability constraint not met")
    else:
        print("Bellman-Ford: It works")
        display_shortest_path(G, path_bellman, bandwidth_matrix, delay_matrix, reliability_matrix)

    path_a_star = a_star_algorithm(G, source, destination)

    bandwidth_constraint, delay_constraint, reliability_constraint = constraints_check(
        path_a_star, bandwidth_matrix, delay_matrix, reliability_matrix
    )

    if not (bandwidth_constraint and delay_constraint and reliability_constraint):
        print("A*: Path doesn't satisfy constraints:")
        if not bandwidth_constraint:
            print("- Bandwidth constraint not met")
        if not delay_constraint:
            print("- Delay constraint not met")
        if not reliability_constraint:
            print("- Reliability constraint not met")
    else:
        print("A*: It works")
        display_shortest_path(G, path_a_star, bandwidth_matrix, delay_matrix, reliability_matrix)

    path_ant_colony = ant_colony_algorithm(G, source, destination)

    bandwidth_constraint, delay_constraint, reliability_constraint = constraints_check(
        path_ant_colony, bandwidth_matrix, delay_matrix, reliability_matrix
    )

    if not (bandwidth_constraint and delay_constraint and reliability_constraint):
        print("Ant Colony Optimization: Path doesn't satisfy constraints:")
        if not bandwidth_constraint:
            print("- Bandwidth constraint not met")
        if not delay_constraint:
            print("- Delay constraint not met")
        if not reliability_constraint:
            print("- Reliability constraint not met")
    else:
        print("Ant Colony Optimization: It works")
        display_shortest_path(G, path_ant_colony, bandwidth_matrix, delay_matrix, reliability_matrix)

    return path_dijkstra, path_bellman, path_a_star, path_ant_colony

# Example usage:
file_path = '/Users/ataberk/PycharmProjects/ataAlgoAnalysis/text.txt'
source_node = 18
destination_node = 19
bandwidth_req = 5

result_dijkstra, result_bellman, result_a_star, result_ant_colony = Solution(file_path, source_node, destination_node, bandwidth_req)
print("Dijkstra Path:", result_dijkstra)
print("Bellman-Ford Path:", result_bellman)
print("A* Path:", result_a_star)
print("Ant Colony Optimization Path:", result_ant_colony)


'''
Here is a brief explanation:

1-Input Reading:

The read_input_file function reads an input file containing information about the network, including adjacency, bandwidth, delay, and reliability matrices.

2-Path Constraints Checking:

The constraints_check function verifies if a given path satisfies certain constraints, such as minimum bandwidth, maximum delay, and minimum reliability.

3-Shortest Path Algorithms:

Dijkstra's, Bellman-Ford, and A* algorithms (dijkstra_algorithm, bellman_ford_algorithm, a_star_algorithm) are used to find the shortest paths.

4-Ant Colony Optimization Algorithm:

The ant_colony_algorithm function implements the Ant Colony Optimization (ACO) algorithm for finding an optimized path, It iteratively constructs paths using artificial ants and updates pheromone levels on the edges.

5-Path Selection in ACO:

The select_next_node function is part of the ACO algorithm and is responsible for selecting the next node based on pheromone levels and visibility.

6-Pheromone Update in ACO:

The update_pheromones function updates the pheromone levels on the edges based on the paths constructed by the ants.

7-Best Path Update in ACO:

The update_best_path function keeps track of the best path found by the ants during the iterations of the ACO algorithm.

8-Graph Creation:

The create_graph function creates a directed graph using the adjacency matrix.

9-Displaying Shortest Path:

The display_shortest_path function visualizes the graph, highlighting the shortest path based on the algorithm used.

10-Solution Function:

The Solution function integrates the above components to solve the optimization problem using Dijkstra's, Bellman-Ford, A*, and Ant Colony Optimization algorithms. 
It checks whether each algorithm's result satisfies the specified constraints and displays the shortest path if it does.

11-Example Usage:

An example is provided at the end of the code, demonstrating how to use the Solution function with a specific input file, source and destination nodes, and bandwidth requirement.'''



'''Our code above shows the algorithms (Dijkstra's, Bellman-Ford, A*, 
and Ant Colony Optimization) for finding the shortest paths in a network. The code checks the constraints such as 
bandwidth, delay, and reliability for the paths obtained by these algorithms, it also shows everything is 
working fine in the console. THe example can be seen below:

Once the user enters a source_node and destination_node'''
