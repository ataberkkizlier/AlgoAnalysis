Project Details:
Input: US backbone topology which consists of 24 nodes and 43
edges is given as input. The input file contains adjacency,
bandwidth, delay, and reliability matrices. The relationship
between each node in these matrices is separated by “:”. Each
matrix is separated by an empty row.
• In the neighborhood matrix, link distances are uniformly
distributed between 1 and 5.
• The bandwidth matrix uniform distribution is between 3 and
10.
• The delay matrix is uniformly distributed between 1 and 5.
• The reliability matrix is between 0.95 and 0.99.
Request: A request will have source node id, destination node
id, and bandwidth requirement information.

Algorithm: Three of the following algorithms must be used to
solve the problem.
• Dijkstra Algorithm
• Bellman-Ford Algorithm
• A* Algorithm
• Flody-Warshall Algorithm
Additional points will be awarded for solving the problem with any
of the meta-heuristic algorithms specified. 
• Simulated Annealing Algorithm
• Tabu Search Algorithm
• Ant Colony Algorithm
• Bee Colony Algorithm
• Firefly Algorithm

After running the algorithm, it should give all nodes
in the calculated path.
The problem should find a solution according to the objective
function given below.
𝑚𝑖𝑛 (𝑏𝑤 × ∑ 𝛿𝑖𝑗 × 𝑑𝑖𝑠𝑡𝑖𝑗
𝑖𝑗 ∈𝐸
)
In the given equation:
• Graph (G) consists of vertex and edges, G=(V,E). i and j are
two vertices defined in the graph and ij denotes an edge in
the edge set.
bw represents bandwidth demand,
𝛿𝑖𝑗 = {
1, 𝑖𝑓 𝑒𝑑𝑔𝑒 (𝑖,𝑗)𝑖𝑠 𝑢𝑠𝑒𝑑
0, 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒
𝑑𝑖𝑠𝑡𝑖𝑗, edge distances between i and j.
