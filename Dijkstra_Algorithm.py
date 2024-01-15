import numpy as np
import networkx as nx

def read_input_file(file_path):
    """
    Read the input file and parse the matrices.

    Parameters:
    - file_path (str): Path to the input file.

    Returns:
    - Tuple of numpy.ndarray: Adjacency matrix, bandwidth matrix, delay matrix, reliability matrix.
    """
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    adjacency_str, bandwidth_str, delay_str, reliability_str = '\n'.join(lines).split('\n\n')

    adjacency_matrix = np.array([list(map(int, row.split(':'))) for row in adjacency_str.strip().split('\n')])
    bandwidth_matrix = np.array([list(map(int, row.split(':'))) for row in bandwidth_str.strip().split('\n')])
    delay_matrix = np.array([list(map(int, row.split(':'))) for row in delay_str.strip().split('\n')])
    reliability_matrix = np.array([list(map(float, row.split(':'))) for row in reliability_str.strip().split('\n')])

    return adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix

def constraints_check(path, bandwidth_matrix, delay_matrix, reliability_matrix):
    """
    Check constraints for a given path.

    Parameters:
    - path (list): List representing the path.
    - bandwidth_matrix (numpy.ndarray): Matrix representing bandwidth values between nodes.
    - delay_matrix (numpy.ndarray): Matrix representing delay values between nodes.
    - reliability_matrix (numpy.ndarray): Matrix representing reliability values between nodes.

    Returns:
    - Tuple of bool: Bandwidth constraint, delay constraint, reliability constraint.
    """
    if not path:
        return False, False, False  # No path found

    path_edges = list(zip(path[:-1], path[1:]))  # Extract edges from the path

    # Extract bandwidth and reliability values for the edges in the path
    bandwidth_values = [bandwidth_matrix[u][v] for u, v in path_edges]
    reliability_values = [reliability_matrix[u][v] for u, v in path_edges]

    min_bandwidth = np.min(bandwidth_values)
    total_delay = np.sum(delay_matrix[np.array(path)])
    min_reliability = np.min(reliability_values)

    print("Bandwidth values along the path:", bandwidth_values)

    bandwidth_constraint = min_bandwidth == 5
    delay_constraint = total_delay < 40
    reliability_constraint = min_reliability > 0.70

    return bandwidth_constraint, delay_constraint, reliability_constraint

def dijkstra_algorithm(graph, source, destination):
    """
    Find the shortest path using Dijkstra's algorithm.

    Parameters:
    - graph (networkx.Graph): Graph representation.
    - source (int): Starting node.
    - destination (int): Target node.

    Returns:
    - Tuple: Distance and path.
    """
    path = nx.single_source_dijkstra(graph, source, target=destination)
    return path

def bellman_ford_algorithm(graph, source, destination):
    """
    Find the shortest path using Bellman-Ford algorithm.

    Parameters:
    - graph (networkx.Graph): Graph representation.
    - source (int): Starting node.
    - destination (int): Target node.

    Returns:
    - list: Path.
    """
    path = nx.bellman_ford_path(graph, source, target=destination)
    return path

def a_star_algorithm(graph, source, destination):
    """
    Find the shortest path using A* algorithm.

    Parameters:
    - graph (networkx.Graph): Graph representation.
    - source (int): Starting node.
    - destination (int): Target node.

    Returns:
    - list: Path.
    """
    path = nx.astar_path(graph, source, destination)
    return path

def print_a_star_result(result_a_star):
    """
    Print the result of A* algorithm.

    Parameters:
    - result_a_star: Result of A* algorithm.
    """
    print("A* Shortest Path:", result_a_star)

def get_path(pred, source, target):
    """
    Reconstruct the path from predecessors.

    Parameters:
    - pred: Predecessors.
    - source (int): Starting node.
    - target (int): Target node.

    Returns:
    - list: Reconstructed path.
    """
    path = [target]
    while path[-1] != source:
        path.append(pred[source][path[-1]])
    return path[::-1]

def create_graph(adjacency_matrix):
    """
    Create a directed graph using NetworkX.

    Parameters:
    - adjacency_matrix (numpy.ndarray): Adjacency matrix.

    Returns:
    - networkx.DiGraph: Directed graph.
    """
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])  # Adjust weight as needed

    return G

def calculate_objective_function(graph, path, bandwidth_matrix, delay_matrix):
    """
    Calculate the objective function value for a given path.

    Parameters:
    - graph (networkx.Graph): Graph representation.
    - path (list): List representing the path.
    - bandwidth_matrix (numpy.ndarray): Matrix representing bandwidth values between nodes.
    - delay_matrix (numpy.ndarray): Matrix representing delay values between nodes.

    Returns:
    - float: Objective function value.
    """
    objective_function = np.min([bandwidth_matrix[path[i]][path[i + 1]] * np.sum(delay_matrix[path[i:i + 2]]) for i in range(len(path) - 1)])
    return objective_function

def Solution(file_path, source, destination, bandwidth_requirement):
    """
    Find paths using various algorithms, check constraints, and calculate the objective function.

    Parameters:
    - file_path (str): Path to the input file.
    - source (int): Starting node.
    - destination (int): Target node.
    - bandwidth_requirement (int): Bandwidth requirement.

    Returns:
    - Tuple of lists: Dijkstra's path, Bellman-Ford's path, A* path
    """
    adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix = read_input_file(file_path)

    G = create_graph(adjacency_matrix)

    # Find the shortest path using Dijkstra's algorithm
    distance_dijkstra, path_dijkstra = dijkstra_algorithm(G, source, destination)

    # Checking constraints for the obtained path using Dijkstra's algorithm
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

    # Find the shortest path using Bellman-Ford algorithm
    path_bellman = bellman_ford_algorithm(G, source, destination)

    # Checking constraints for the obtained path using Bellman-Ford algorithm
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

    # Find the shortest path using A* algorithm
    path_a_star = a_star_algorithm(G, source, destination)

    # Checking constraints for the obtained path using A* algorithm
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

    # Calculate objective function values for each path
    obj_func_dijkstra = calculate_objective_function(G, path_dijkstra, bandwidth_matrix, delay_matrix)
    obj_func_bellman = calculate_objective_function(G, path_bellman, bandwidth_matrix, delay_matrix)
    obj_func_a_star = calculate_objective_function(G, path_a_star, bandwidth_matrix, delay_matrix)

    return path_dijkstra, path_bellman, path_a_star, obj_func_dijkstra, obj_func_bellman, obj_func_a_star

# Example usage:
file_path = '/Users/ataberk/PycharmProjects/ataAlgoAnalysis/text.txt'
source_node = 10
destination_node = 14
bandwidth_req = 5

'''3,4 bandwidth 5 
18,19 bandwidth 5 
2,3 bandwidth 5
10,14 bandwidth 5'''

result_dijkstra, result_bellman, result_a_star, obj_func_dijkstra, obj_func_bellman, obj_func_a_star = Solution(file_path, source_node, destination_node, bandwidth_req)

print("Dijkstra Path:", result_dijkstra)
print("Nodes in Dijkstra Path:", result_dijkstra)
print("Bellman-Ford Path:", result_bellman)
print("Nodes in Bellman-Ford Path:", result_bellman)
print("A* Path:", result_a_star)
print("Nodes in A* Path:", result_a_star)


'''
1-Input File Reading:

The read_input_file function reads an input file containing four matrices
 (adjacency, bandwidth, delay, reliability) and returns them as NumPy arrays.

2-Constraint Checking:

The constraints_check function takes a path and checks whether 
it satisfies specified constraints based on bandwidth, delay, and reliability matrices.

3-Graph Creation:

The create_graph function constructs a directed graph using NetworkX based on the provided adjacency matrix.

4-Pathfinding Algorithms:

Dijkstra's algorithm (dijkstra_algorithm), Bellman-Ford algorithm (bellman_ford_algorithm), 
and A* algorithm (a_star_algorithm) are used to find paths in the graph.

5-Objective Function Calculation:

The calculate_objective_function function computes an objective function value
or a given path using bandwidth and delay matrices.

6-Solution Function:

The Solution function integrates the above components. It finds paths using various algorithms, 
checks constraints, and calculates objective function values.

7-Example Usage:

An example usage is provided at the end, demonstrating how to use the Solution function with 
a sample file path, source and destination nodes, and bandwidth requirement.

8-Printed Output:

The script prints information about each algorithm's path, checks whether the paths satisfy constraints, and calculates and prints objective function value'''



