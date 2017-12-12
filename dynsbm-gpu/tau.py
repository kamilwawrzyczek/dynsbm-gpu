import tensorflow as tf

from graph_data import GraphData
from utils import reverse_01_values, fix_numeric_issues


class Tau:
    def __init__(self, data: GraphData, clusters: [int]):
        tau1_init = self.__init_tau1(data.N, data.Q, clusters)
        taut_init = self.__init_taut(data.T, data.N, data.Q, clusters)

        self.tau1_value = tau1_init
        self.taum_value = None

        self.tau1 = tf.Variable(tau1_init, dtype=tf.float64)
        self.taut = tf.Variable(taut_init, dtype=tf.float64)
        self.taum = tf.Variable(tf.zeros([data.T, data.N, data.Q], dtype=tf.float64))

        self.__update_tau1 = None
        self.__update_taut = None
        self.__update_taum = None

        self.__assign_tau1 = None
        self.__assign_taut = None
        self.__assign_taum = None

    def init(self, data: GraphData, density: tf.Variable, stationary: tf.Variable, trans: tf.Variable):
        self.__update_tau1 = self.__generate_update_tau1_graph(data, density, stationary)
        self.__update_taut = self.__generate_update_taut_graph(data, density, trans)
        self.__update_taum = self.__generate_update_taum_graph(data.T)

        self.__assign_tau1 = tf.assign(self.tau1, self.__update_tau1)
        self.__assign_taut = tf.assign(self.taut, self.__update_taut)
        self.__assign_taum = tf.assign(self.taum, self.__update_taum)

    def update_taut(self, session: tf.Session):
        return session.run(self.__assign_taut)

    def update_tau1(self, session: tf.Session):
        self.tau1_value = session.run(self.__assign_tau1)
        return self.tau1_value

    def update_taum(self, session: tf.Session):
        self.taum_value = session.run(self.__assign_taum)
        return self.taum_value

    def update_tau(self, session: tf.Session):
        return [self.update_tau1(session), self.update_taut(session), self.update_taum(session)]

    def __init_tau1(self, N, Q, clusters):
        tau1 = [[0 for i in range(Q)] for j in range(N)]
        for i in range(N):
            tau1[i][clusters[i] - 1] = 1
        return self.__fix_tau1(tau1, N, Q)

    @staticmethod
    def __fix_tau1(tau1, N, Q):
        for i in range(N):
            for q in range(Q):
                if tau1[i][q] < 1e-7:
                    tau1[i][q] = 1e-7
            array_sum = sum(tau1[i])
            tau1[i] = [x / array_sum for x in tau1[i]]
        return tau1

    def __init_taut(self, T, N, Q, clusters):
        taut = [[[[0 for a in range(Q)] for b in range(Q)] for c in range(N)] for d in range(T - 1)]
        for t in range(T - 1):
            for i in range(N):
                for q in range(Q):
                    taut[t][i][q][clusters[i] - 1] = 1
        return self.__fix_taut(taut, T, N, Q)

    @staticmethod
    def __fix_taut(taut, T, N, Q):
        for t in range(T - 1):
            for i in range(N):
                for q in range(Q):
                    for qp in range(Q):
                        if taut[t][i][q][qp] < 1e-7:
                            taut[t][i][q][qp] = 1e-7
                    array_sum = sum(taut[t][i][q])
                    taut[t][i][q] = [x / array_sum for x in taut[t][i][q]]
        return taut

    def __generate_update_taum_graph(self, T):
        new_taum = tf.expand_dims(self.tau1, axis=0)
        for i in range(T - 1):
            einsum = tf.einsum("iw,iwq->iq", new_taum[i], self.taut[i])
            new_taum = tf.concat([new_taum, tf.expand_dims(einsum, axis=0)], axis=0)

        return fix_numeric_issues(new_taum, above=False, normalize=True)

    def __generate_update_tau1_graph(self, data: GraphData, density: tf.Variable, stationary):
        N = data.N
        Q = data.Q
        K = data.K

        data_correction_array = tf.one_hot(tf.reshape(data.graph[0], [-1]), K + 1, dtype=tf.float64)
        data_correction_array = tf.reshape(data_correction_array, [N, N, 1, 1, K + 1])
        correction_i_eq_j = reverse_01_values(tf.one_hot(tf.range(0, N), N, dtype=tf.float64))
        correction_i_eq_j = tf.reshape(correction_i_eq_j, [N, N, 1, 1, 1])

        taul_reshaped = tf.reshape(self.tau1, [1, N, 1, Q, 1])
        density_reshaped = tf.reshape(density[0], [1, 1, Q, Q, K + 1])
        logp = tf.reduce_sum(taul_reshaped * density_reshaped * data_correction_array * correction_i_eq_j, [1, 3, 4])
        tau1i = logp + tf.reshape(tf.log(stationary), [1, Q])
        # normalization
        tau1i = tf.exp(tau1i - tf.reduce_max(tau1i))

        tau1i_sum = tf.reduce_sum(tau1i, [1], keep_dims=True)
        tau1i = tau1i / tau1i_sum

        tau1 = fix_numeric_issues(tau1i, above=False, normalize=True)
        return tau1

    def __generate_update_taut_graph(self, data: GraphData, density: tf.Variable, trans: tf.Variable):
        T = data.T
        N = data.N
        Q = data.Q
        K = data.K

        data_correction_array = tf.one_hot(tf.reshape(data.graph, [-1]), K + 1, dtype=tf.float64)
        data_correction_array = tf.reshape(data_correction_array, [T, N, 1, N, 1, K + 1])
        correction_i_eq_j = reverse_01_values(tf.one_hot(tf.range(0, N), N, dtype=tf.float64))
        correction_i_eq_j = tf.reshape(correction_i_eq_j, [1, N, 1, N, 1, 1])

        # t, i, q, qprime, j, l, k
        taum_reshaped = tf.reshape(self.taum, [T, 1, 1, N, Q, 1])
        taum_reshaped = tf.tile(taum_reshaped, [1, N, Q, 1, 1, K + 1])
        density_reshaped = tf.reshape(density, [T, 1, Q, 1, Q, K + 1])

        logp = taum_reshaped * density_reshaped * data_correction_array * correction_i_eq_j
        logp = tf.reduce_sum(logp, [3, 4, 5])
        tauti = tf.reshape(logp, [T, N, 1, Q]) + tf.reshape(tf.log(trans), [1, 1, Q, Q])
        tauti = tf.exp(tauti - tf.reduce_max(tauti))

        tauti_sum = tf.reduce_sum(tauti, [3], keep_dims=True)
        taut = tauti / tauti_sum
        taut = tf.slice(taut, [1, 0, 0, 0], [-1, -1, -1, -1])
        taut = fix_numeric_issues(taut, above=False, normalize=True)
        return taut
