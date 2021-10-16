import tensorflow as tf

a = tf.constant([1,2,3], dtype=tf.int64)
a = tf.data.Dataset.from_tensor_slices(a)

