import random
import time

import tensorflow as tf

from graph_data import GraphData
from k_means import k_means_clustering


def find_random_initial_clustering(data: GraphData, seed=time.time()):
    random.seed(seed)
    # return random assignment individuals to groups
    return [random.randint(1, data.Q) for i in range(data.N)]


def find_initial_clustering(data: GraphData):
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as session:
        # create data matrix containing all the adjacency time step matrices stacked in consecutive column blocks
        all_graph_data_matrix = tf.reshape(tf.transpose(data.graph, [1, 0, 2]), [data.N, data.T * data.N])

        # run k-means algorithm and return individuals assignment to groups
        _, assignment_values = k_means_clustering(session, tf.to_float(all_graph_data_matrix), data.Q)
        return assignment_values + 1
