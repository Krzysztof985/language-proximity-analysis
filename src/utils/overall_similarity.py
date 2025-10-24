def add_connection(graph, node1, node2, label):
    # Node adding
    graph.add_node(node1)
    graph.add_node(node2)
    # Edge adding
    graph.add_edge(node1, node2, label=label)


def diagonal_average(matrix):
    """
    Returns the diagonal average of the matrix
    """
    n = min(len(matrix), len(matrix[0]))  # in case of non-square matrix
    diagonal = [matrix[i][i] for i in range(n)]
    return sum(diagonal) / len(diagonal) if diagonal else 0
