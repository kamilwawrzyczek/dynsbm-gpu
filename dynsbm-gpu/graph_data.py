import tensorflow as tf


class GraphData:
    def __init__(self, T, N, Q, K, graph) -> None:
        self.T = T
        self.N = N
        self.Q = Q
        self.K = K
        self.graph = tf.constant(graph)

    @staticmethod
    def load_graph_data(file_name, separator=','):
        with open(file_name, 'r') as data_file:
            first_line = data_file.readline().replace("\r\n", "")
            first_line_split = first_line.split(separator)
            T = int(first_line_split[0])
            N = int(first_line_split[1])
            Q = int(first_line_split[2])
            K = 0
            graph_matrix = [[[0 for a in range(N)] for b in range(N)] for c in range(T)]
            for line in data_file:
                line_split = line.replace("\r\n", "").replace("\n", "").split(separator)
                if line_split.__len__() != 4:
                    continue
                t = int(line_split[0]) - 1
                x = int(line_split[1]) - 1
                y = int(line_split[2]) - 1
                value = int(line_split[3])
                K = max(K, value)
                graph_matrix[t][x][y] = value
            return GraphData(T, N, Q, K, graph_matrix)
