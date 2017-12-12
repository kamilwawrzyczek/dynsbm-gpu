import tensorflow as tf

from graph_data import GraphData
from tau import Tau
from utils import reverse_01_values, fix_numeric_issues


class Multinomial:
    def __init__(self, data: GraphData, tau: Tau):
        self.density = tf.Variable(tf.zeros([data.T, data.Q, data.Q, data.K + 1], dtype=tf.float64))
        self.__update_density = self.__prepare_update_density(data, tau)
        self._assign_op = tf.assign(self.density, self.__update_density)

    def update_density(self, session: tf.Session):
        return session.run(self._assign_op)

    def __prepare_update_density(self, data: GraphData, tau: Tau):
        T = data.T
        Q = data.Q
        N = data.N
        K = data.K

        correction_ij_array = tf.sequence_mask(tf.range(0, N), N, dtype=tf.float64)
        correction_ql_array = tf.sequence_mask(tf.range(0, Q), Q, dtype=tf.float64)

        # betaql is calculated when data is eq 0, transform data into array with 1 when data eq 0 and 0 otherwise
        data_boolean_mask = tf.cast(tf.cast(data.graph, dtype=tf.bool), dtype=tf.float64)
        data_correction_array = reverse_01_values(data_boolean_mask)

        correction_array_1 = tf.reshape(correction_ij_array, [1, N, N, 1, 1]) * \
                             tf.reshape(correction_ql_array, [1, 1, 1, Q, Q]) * \
                             tf.reshape(data_correction_array, [T, N, N, 1, 1])

        correction_array_2 = tf.reshape(correction_ij_array, [1, N, N]) * data_correction_array
        betaql = self.__betaql(data, tau, correction_array_1, correction_array_2)

        # calculate betaqlsum
        correction_array_1 = tf.reshape(correction_ij_array, [N, N, 1, 1]) * \
                             tf.reshape(correction_ql_array, [1, 1, Q, Q])
        correction_array_1 = tf.reshape(correction_array_1, [1, N, N, Q, Q])

        correction_array_2 = tf.expand_dims(correction_ij_array, 0)
        betasumql = self.__betaql(data, tau, correction_array_1, correction_array_2)
        betaql = tf.divide(betaql, betasumql)
        betaql = fix_numeric_issues(betaql)

        # calculate multinmprobaql
        data_correction_array = tf.one_hot(tf.reshape(data.graph - 1, [-1]), K, dtype=tf.float64)
        data_correction_array = tf.reshape(data_correction_array, [T, N, N, K])

        correction_array_1 = tf.multiply(tf.reshape(correction_ij_array, [1, N, N, 1]), data_correction_array)
        correction_array_1 = tf.multiply(tf.reshape(correction_array_1, [T, N, N, 1, 1, K]),
                                         tf.reshape(correction_ql_array, [1, 1, 1, Q, Q, 1]))

        multinomprobaql_1a = tf.reshape(tau.taum, [T, N, 1, Q, 1, 1]) * \
                             tf.reshape(tau.taum, [T, 1, N, 1, Q, 1]) * correction_array_1
        multinomprobaql_1a = tf.reduce_sum(multinomprobaql_1a, [1, 2])

        multinomprobaql_1b = tf.reshape(tau.taum, [T, N, 1, 1, Q, 1]) * \
                             tf.reshape(tau.taum, [T, 1, N, Q, 1, 1]) * correction_array_1
        multinomprobaql_1b = tf.reduce_sum(multinomprobaql_1b, [1, 2])

        multinomprobaql_1 = multinomprobaql_1a + multinomprobaql_1b
        multinomprobaql_1 = tf.reduce_sum([multinomprobaql_1, tf.transpose(multinomprobaql_1, [0, 2, 1, 3])], 0)

        correction_array_2 = tf.reshape(correction_ij_array, [1, N, N, 1]) * data_correction_array

        multinomprobaql_2 = tf.reshape(tau.taum, [T, N, 1, Q, 1]) * tf.reshape(tau.taum, [T, 1, N, Q, 1]) * \
                            tf.reshape(correction_array_2, [T, N, N, 1, K])
        multinomprobaql_2 = tf.reduce_sum(multinomprobaql_2, [0, 1, 2])

        mask = tf.one_hot(tf.range(0, Q), Q, dtype=tf.float64)
        # multinomprobaql_2 = tf.tile(tf.reshape(multinomprobaql_2, [Q, 1, K]), [1, Q, 1])
        multinomprobaql_2 = tf.reshape(multinomprobaql_2, [Q, 1, K])
        multinomprobaql_2 = multinomprobaql_2 * tf.reshape(mask, [Q, Q, 1])
        # multinomprobaql_2 = tf.tile([multinomprobaql_2], [T, 1, 1, 1])
        multinomprobaql_2 = tf.expand_dims(multinomprobaql_2, 0)

        multinomprobaql = multinomprobaql_1 + multinomprobaql_2
        multinomprobaql = multinomprobaql / tf.reduce_sum(multinomprobaql, 3, keep_dims=True)
        multinomprobaql = fix_numeric_issues(multinomprobaql)

        betaql_log = tf.log(betaql)
        density_1 = tf.pad(tf.reshape(betaql_log, [T, Q, Q, 1]), [[0, 0], [0, 0], [0, 0], [0, K]])

        betaql_log = tf.log(tf.subtract(tf.constant(1, dtype=tf.float64), betaql))
        multinomprobaql_log = tf.log(multinomprobaql)

        density_2 = tf.add(tf.reshape(betaql_log, [T, Q, Q, 1]), multinomprobaql_log)

        density = tf.add(density_1, tf.pad(density_2, [[0, 0], [0, 0], [0, 0], [1, 0]]))
        return density

    def __betaql(self, data: GraphData, tau: Tau, correction_array_1, correction_array_2):
        T = data.T
        N = data.N
        Q = data.Q
        betaql_1a = tf.reshape(tau.taum, [T, N, 1, Q, 1]) * tf.reshape(tau.taum, [T, 1, N, 1, Q]) * correction_array_1
        betaql_1a = tf.reduce_sum(betaql_1a, [1, 2])

        betaql_1b = tf.reshape(tau.taum, [T, N, 1, 1, Q]) * tf.reshape(tau.taum, [T, 1, N, Q, 1]) * correction_array_1
        betaql_1b = tf.reduce_sum(betaql_1b, [1, 2])

        betaql_1 = betaql_1a + betaql_1b
        betaql_1 = betaql_1 + tf.transpose(betaql_1, [0, 2, 1])

        # calculate betaql for q=l -> t=1 gathers all t for q=l
        betaql_2 = tf.reshape(tau.taum, [T, N, 1, Q]) * tf.reshape(tau.taum, [T, 1, N, Q]) * \
                   tf.expand_dims(correction_array_2, [-1])
        betaql_2 = tf.reduce_sum(betaql_2, [0, 1, 2])
        betaql_2 = tf.expand_dims(tf.diag(betaql_2), 0)

        betaql = betaql_1 + betaql_2
        return betaql
