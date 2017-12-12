# K-Means clustering from https://gist.github.com/narphorium/d06b7ed234287e319f18 which some small modifications
import tensorflow as tf


def k_means_clustering(sess: tf.Session, vectors, num_clusters, max_num_steps=100, stop_coeficient=0.0):
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                     [0, 0], [num_clusters, -1]))
    old_centroids = tf.Variable(tf.zeros(tf.shape(centroids)))
    centroid_distance = tf.Variable(tf.zeros(tf.shape(centroids)))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(
        tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    means = tf.concat([
        tf.reduce_mean(
            tf.gather(vectors,
                      tf.reshape(
                          tf.where(
                              tf.equal(assignments, c)
                          ), [1, -1])
                      ), reduction_indices=[1])
        for c in range(num_clusters)], 0)

    save_old_centroids = tf.assign(old_centroids, centroids)

    update_centroids = tf.assign(centroids, means)
    init_op = tf.global_variables_initializer()

    performance = tf.assign(centroid_distance, tf.subtract(centroids, old_centroids))
    check_stop = tf.reduce_sum(tf.abs(performance))

    sess.run(init_op)
    for step in range(max_num_steps):
        sess.run(save_old_centroids)
        _, centroid_values, assignment_values = sess.run([update_centroids,
                                                          centroids,
                                                          assignments])
        sess.run(check_stop)
        current_stop_coeficient = check_stop.eval()
        if current_stop_coeficient <= stop_coeficient:
            break

    return centroid_values, assignment_values
