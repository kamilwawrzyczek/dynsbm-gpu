import tensorflow as tf

from clustering import calculate_clusters

gpu = True
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

# data_file = "data/12_100_20.txt"
# data_file = "data/12_100_5.txt"
# data_file = "data/12_50_20_5.txt"
data_file = "data/12_50_20.txt"
# data_file = "data/12_50_10.txt"
# data_file = "data/12_40_20_5.txt"
# data_file = "data/12_10_2_5.txt"
# data_file = "data/example_data_1.csv"

results, measurements = calculate_clusters(data_file_name=data_file, use_gpu=gpu,
                                           init_method="random", config=config,
                                           max_iterations=max_iterations,
                                           max_internal_iterations=max_iterations)

print("Device: ", "GPU" if gpu else "CPU")
print("Data file:", data_file)
print()
print("Elapsed time: ", measurements.time)
print("Iterations (internal): ", measurements.iterations, "(", measurements.internal_iterations, ")")
print("Memory used (CPU): ", measurements.cpu_memory)
print("Memory used (GPU): ", measurements.gpu_memory)
print()
print("Log likelihood: ", results.likelihood)
print("Membership", results.memberships)
print("Trans:", results.transitions)
