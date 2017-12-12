import time

import numpy as np
import psutil

from init_graph import *
from likelihood import Likelihood
from markov_chain import MarkovChain
from multinomial import Multinomial
from tau import Tau


class Results:
    def __init__(self, likelihood, trans, memberships, initial_clusters) -> None:
        self.likelihood = likelihood
        self.transitions = trans
        self.memberships = memberships
        self.initial_clusters = initial_clusters


class Measurements:
    def __init__(self, generate_time, execution_time, cpu_memory, gpu_memory, iterations, internal_iterations,
                 loglikelihood_history, N):
        self.N = N
        self.loglikelihood_history = loglikelihood_history
        self.generate_time = generate_time
        self.execution_time = execution_time
        self.cpu_memory = cpu_memory
        self.gpu_memory = gpu_memory
        self.iterations = iterations
        self.internal_iterations = internal_iterations


def calculate_clusters(data_file_name, init_method="kmeans", max_iterations=10, max_internal_iterations=10,
                       use_gpu=False, config: tf.ConfigProto = tf.ConfigProto()):
    start = time.time()
    iterations = 0
    internal_iterations = 0
    loglikelihood_history = []
    with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
        data = GraphData.load_graph_data(data_file_name)

        if init_method == "random":
            initial_clusters = find_random_initial_clustering(data)
        else:
            initial_clusters = find_initial_clustering(data)

        tau = Tau(data, initial_clusters)
        multinomial = Multinomial(data, tau)
        markov_chain = MarkovChain(data, tau)
        likelihood = Likelihood(data, tau, multinomial, markov_chain)
        tau.init(data, multinomial.density, markov_chain.stationary, markov_chain.trans)
        generate_time_end = time.time()

        likelihood_result = None
        trans_result = None
        with tf.Session(config=config) as session:
            # init
            session.run(tf.global_variables_initializer())
            tau.update_taum(session)
            multinomial.update_density(session)

            previous_log = -1e30
            for i in range(max_iterations):
                iterations += 1
                previous_log_internal = -1e30
                for z in range(max_internal_iterations):
                    internal_iterations += 1
                    tau.update_tau(session)
                    new_log = likelihood.calculate(session)
                    if abs(previous_log_internal - new_log) < 1e-7:
                        break
                    loglikelihood_history.append(new_log)
                    previous_log_internal = new_log

                trans_result = markov_chain.update_trans(session)
                markov_chain.update_stationary(session)
                multinomial.update_density(session)

                new_log = likelihood.calculate(session)
                likelihood_result = new_log
                loglikelihood_history.append(likelihood_result)
                if previous_log > new_log:
                    break
                previous_log = new_log

            memory_gpu = 0
            if use_gpu:
                memory_gpu = session.run(tf.contrib.memory_stats.MaxBytesInUse()) / 1024 / 1024

    end = time.time()
    membership = __find_element_membership(data, tau)
    memory_cpu = psutil.Process().memory_info().rss / 1024 / 1024

    results = Results(likelihood_result, trans_result, membership, initial_clusters)
    measurements = Measurements(generate_time_end - start, end - generate_time_end, memory_cpu, memory_gpu, iterations,
                                internal_iterations, loglikelihood_history, data.N)
    return results, measurements


def __find_element_membership(data: GraphData, tau: Tau):
    return [__find_element_membership_t(t, data.N, tau.tau1_value, tau.taum_value) for t in range(data.T)]


def __find_element_membership_t(t, N, tau1, taum):
    if t == 0:
        return [np.argmax(tau1[i]) + 1 for i in range(N)]
    else:
        return [np.argmax(taum[t, i]) + 1 for i in range(N)]
