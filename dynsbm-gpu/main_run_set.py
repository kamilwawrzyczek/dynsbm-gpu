import sys

import tensorflow as tf

from clustering import calculate_clusters

max_iterations = 10
max_cpu_threads = 1
no_opt = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                             do_common_subexpression_elimination=True,
                             do_function_inlining=True,
                             do_constant_folding=True)
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=no_opt),
                        log_device_placement=False, allow_soft_placement=True)
config.intra_op_parallelism_threads = max_cpu_threads
config.inter_op_parallelism_threads = max_cpu_threads

data_file = str(sys.argv[1])
gpu = bool(sys.argv[2])
init_method = str(sys.argv[3])
iteration = int(sys.argv[4])

print("-----------------------------------------------------------------------", file=sys.stderr)
print(data_file, iteration, init_method, file=sys.stderr)
print("-----------------------------------------------------------------------", file=sys.stderr)

results, measurements = calculate_clusters(data_file_name=data_file, use_gpu=gpu,
                                           init_method=init_method, config=config,
                                           max_iterations=max_iterations,
                                           max_internal_iterations=max_iterations)

print(data_file, iteration, "GPU" if gpu else "CPU", init_method, measurements.N, measurements.generate_time,
      measurements.execution_time, measurements.generate_time + measurements.execution_time, measurements.iterations,
      measurements.internal_iterations, measurements.cpu_memory, measurements.gpu_memory, results.likelihood,
      measurements.loglikelihood_history, sep="|")
