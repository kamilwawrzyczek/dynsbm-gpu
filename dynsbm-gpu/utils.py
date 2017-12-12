import tensorflow as tf


def reverse_01_values(correction_ij_array):
    return (correction_ij_array - 1) * (-1)


def fix_numeric_issues(data, below=True, above=True, normalize=False):
    if below:
        data = tf.maximum(data, tf.cast(tf.fill(tf.shape(data), 1e-7), dtype=tf.float64))
    if above:
        data = tf.minimum(data, tf.cast(tf.fill(tf.shape(data), 1 - 1e-7), dtype=tf.float64))
    if normalize:
        data_sum = tf.reduce_sum(data, [-1])
        data_sum = tf.expand_dims(data_sum, [-1])
        data = data / data_sum
    return data
