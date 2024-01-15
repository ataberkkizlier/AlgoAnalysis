import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
        return False, False, False  # No path found

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
    path = nx.single_source_dijkstra(graph, source, target=destination)[1]
    return path


def bellman_ford_algorithm(graph, source, destination):
    path = nx.bellman_ford_path(graph, source, target=destination)
    return path


def a_star_algorithm(graph, source, destination):
    path = nx.astar_path(graph, source, destination)
    return path


def visualize_graph(graph, title):
    pos = nx.spring_layout(graph)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=400, node_color='skyblue')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.title(title)
    plt.show()


def create_graph(adjacency_matrix):
    G = nx.DiGraph()
    num_nodes = len(adjacency_matrix)
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                G.add_edge(i, j, weight=adjacency_matrix[i][j])

    return G


def Solution(file_path, source, destination, bandwidth_requirement):
    adjacency_matrix, bandwidth_matrix, delay_matrix, reliability_matrix = read_input_file(file_path)

    G = create_graph(adjacency_matrix)

    print("Original Graph:")
    visualize_graph(G, "Original Graph")

    # Find the shortest path using Dijkstra's algorithm
    distance_dijkstra, path_dijkstra = dijkstra_algorithm(G, source, destination)

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

    # Visualize the graph after Dijkstra's algorithm
    visualize_graph(G, "Graph after Dijkstra's Algorithm")

    # Create a new graph to preserve the original state
    G_bellman = create_graph(adjacency_matrix)

    # Find the shortest path using Bellman-Ford algorithm
    path_bellman = bellman_ford_algorithm(G_bellman, source, destination)

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

    # Visualize the graph after Bellman-Ford algorithm
    visualize_graph(G_bellman, "Graph after Bellman-Ford Algorithm")

    # Create another new graph to preserve the original state
    G_a_star = create_graph(adjacency_matrix)

    # Find the shortest path using A* algorithm
    path_a_star = a_star_algorithm(G_a_star, source, destination)

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

    # Visualize the final graph after A* algorithm
    visualize_graph(G_a_star, "Graph after A* Algorithm")

    return G, G_bellman, G_a_star, path_dijkstra, path_bellman, path_a_star


# Example usage:
file_path = '/Users/ataberk/PycharmProjects/ataAlgoAnalysis/text.txt'
source_node = 3
destination_node = 4
bandwidth_req = 5

G, G_bellman, G_a_star, result_dijkstra, result_bellman, result_a_star = Solution(file_path, source_node,
                                                                                  destination_node, bandwidth_req)
print("Dijkstra Path:", result_dijkstra)
print("Bellman-Ford Path:", result_bellman)
print("A* Path:", result_a_star)
