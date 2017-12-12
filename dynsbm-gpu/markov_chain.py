import tensorflow as tf

from graph_data import GraphData
from tau import Tau
from utils import fix_numeric_issues


class MarkovChain:
    def __init__(self, data: GraphData, tau: Tau):
        fill_data = tf.constant(1 / data.Q, dtype=tf.float64)
        self.stationary = tf.Variable(tf.fill([data.Q], fill_data))
        self.trans = tf.Variable(tf.fill([data.Q, data.Q], fill_data))
        self.__update_trans = self.__generate_update_trans_graph(data, tau)
        self.__update_stationary = self.__generate_update_stationary_graph(tau)
        self.__assign_trans = tf.assign(self.trans, self.__update_trans)
        self.__assign_stationary = tf.assign(self.stationary, self.__update_stationary)

    def update_trans(self, session: tf.Session):
        return session.run(self.__assign_trans)

    def update_stationary(self, session: tf.Session):
        return session.run(self.__assign_stationary)

    @staticmethod
    def __generate_update_trans_graph(data: GraphData, tau: Tau):
        T = data.T
        N = data.N
        Q = data.Q

        taum_slice = tf.slice(tau.taum, [0, 0, 0], [T - 1, -1, -1])
        reshaped_taum = tf.reshape(taum_slice, [T - 1, N, Q, 1])
        trans_1 = tau.taut * reshaped_taum
        trans = tf.reduce_sum(trans_1, [0, 1])

        trans = fix_numeric_issues(trans, above=False, normalize=True)
        return trans

    @staticmethod
    def __generate_update_stationary_graph(tau: Tau):
        stationary = tf.reduce_sum(tau.taum, [0, 1])
        stationary = fix_numeric_issues(stationary, above=False)

        stationary_sum = tf.reduce_sum(stationary)
        stationary = stationary / stationary_sum
        return stationary
