import tensorflow as tf

from graph_data import GraphData
from markov_chain import MarkovChain
from multinomial import Multinomial
from tau import Tau


class Likelihood:
    def __init__(self, data: GraphData, tau: Tau, multinomial: Multinomial, markov: MarkovChain):
        self.__sum = self.__prepare_graph(data, tau, multinomial, markov)

    @staticmethod
    def __prepare_graph(data: GraphData, tau: Tau, multinomial: Multinomial, markov: MarkovChain):
        T = data.T
        N = data.N
        Q = data.Q
        K = data.K

        # calculate sum 1
        log_stationary = tf.reshape(tf.tile(tf.log(markov.stationary), [N]), [N, Q])
        log_tau1 = tf.negative(tf.log(tau.tau1))

        reduced_sum1 = tf.reduce_sum([log_stationary, log_tau1], 0)
        reduced_prod1 = tf.reduce_prod([tau.tau1, reduced_sum1], 0)
        sum1 = tf.reduce_sum(reduced_prod1)

        # calculate sum 2
        log_trans = tf.reshape(tf.log(markov.trans), [1, 1, Q, Q])
        log_taut = tf.log(tau.taut)
        reduced_sum2 = log_trans - log_taut

        # get [0..T-1][0..N][0..Q] slice
        marginal_reshaped = tf.slice(tau.taum, [0, 0, 0], [T - 1, -1, -1])
        marginal_reshaped = tf.reshape(marginal_reshaped, [T - 1, N, Q, 1])

        reduced_prod2 = marginal_reshaped * tau.taut * reduced_sum2
        sum2 = tf.reduce_sum(reduced_prod2)

        # calculate sum 3
        data_correction_array = tf.one_hot(tf.reshape(data.graph, [-1]), K + 1, dtype=tf.float64)
        data_correction_array = tf.reshape(data_correction_array, [T, N, N, K + 1])
        correction_ij_array = tf.sequence_mask(tf.range(0, N), N, dtype=tf.float64)

        marginal1 = tf.reshape(tau.taum, [T, N, 1, Q, 1])
        marginal2 = tf.reshape(tau.taum, [T, 1, N, 1, Q])
        marginal = tf.multiply(marginal1, marginal2)
        marginal = tf.multiply(marginal, tf.reshape(correction_ij_array, [1, N, N, 1, 1]))

        marginal = tf.reshape(marginal, [T, N, N, Q, Q, 1])
        marginal = tf.multiply(marginal, tf.reshape(data_correction_array, [T, N, N, 1, 1, K + 1]))
        marginal = tf.multiply(marginal, tf.reshape(multinomial.density, [T, 1, 1, Q, Q, K + 1]))

        sum3 = tf.reduce_sum(marginal)
        return tf.reduce_sum([sum1, sum2, sum3])

    def calculate(self, session: tf.Session):
        return session.run(self.__sum)
