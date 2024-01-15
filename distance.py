'''DISTANCE GRAPH'''


''' Added the matrix which contains the distances between nodes, we used a graph 
using matplotlib for the user to see it better visually'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


distance_matrix_str = """
0:3:0:0:0:1:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
3:0:3:0:0:5:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
0:3:0:5:2:0:5:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
0:0:5:0:5:0:4:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
0:0:2:5:0:0:0:4:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
1:5:0:0:0:0:3:0:3:0:2:0:0:0:0:0:0:0:0:0:0:0:0:0
0:0:5:4:0:3:0:1:2:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0
0:0:0:0:4:0:1:0:0:2:0:0:0:0:0:0:0:0:0:0:0:0:0:0
0:0:0:0:0:3:2:0:0:2:3:3:0:0:0:0:0:0:0:0:0:0:0:0
0:0:0:0:0:0:0:2:2:0:0:0:5:4:0:0:0:0:0:0:0:0:0:0
0:0:0:0:0:2:0:0:3:0:0:1:0:0:4:0:0:0:2:0:0:0:0:0
0:0:0:0:0:0:0:0:3:0:1:0:3:0:0:2:0:0:0:0:0:0:0:0
0:0:0:0:0:0:0:0:0:5:0:3:0:2:0:0:3:0:0:0:0:0:0:0
0:0:0:0:0:0:0:0:0:4:0:0:2:0:0:0:0:5:0:0:0:0:0:0
0:0:0:0:0:0:0:0:0:0:4:0:0:0:0:3:0:0:0:4:0:0:0:0
0:0:0:0:0:0:0:0:0:0:0:2:0:0:3:0:4:0:0:0:5:2:0:0
0:0:0:0:0:0:0:0:0:0:0:0:3:0:0:4:0:4:0:0:0:1:4:0
0:0:0:0:0:0:0:0:0:0:0:0:0:5:0:0:4:0:0:0:0:0:0:4
0:0:0:0:0:0:0:0:0:0:2:0:0:0:0:0:0:0:0:3:0:0:0:0
0:0:0:0:0:0:0:0:0:0:0:0:0:0:4:0:0:0:3:0:1:0:0:0
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:5:0:0:0:1:0:2:0:0
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:2:1:0:0:0:2:0:3:0
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:4:0:0:0:0:3:0:3
0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:4:0:0:0:0:3:0
"""

# Convert the string matrix to a 2D NumPy array
distance_matrix = np.array([list(map(int, row.split(':'))) for row in distance_matrix_str.strip().split('\n')])

# Create a graph using NetworkX
G = nx.Graph()

# Add edges to the graph
num_nodes = len(distance_matrix)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        weight = distance_matrix[i][j]
        if weight > 0:
            G.add_edge(i, j, weight=weight)


# Drawing the graph step 1:

pos = nx.spring_layout(G, seed=42)
labels = {i: i for i in G.nodes()}  # Use node indices as labels

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, labels=labels, font_weight='bold', node_size=300, node_color='skyblue', font_color='black', font_size=8, edge_color='gray', width=1.5)

# Draw edge labels (distances)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Added a title to the graph.
plt.suptitle("Distance Graph", y=0.95, fontsize=14, fontweight='bold', color="red")

# Show the graph
plt.show()
