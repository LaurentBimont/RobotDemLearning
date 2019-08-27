import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

output = np.array([[1, 0, 2], [0, 1, 1]])
label = np.array([[1, 0, 2], [0, 1, 1]])


# flat_labels = tf.reshape(tensor=label, shape=(-1, 3))
#
# flat_logits = tf.reshape(tensor=logits, shape=(-1, 3))

cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label)
sum_cross = tf.reduce_sum(cross_entropies)

print(cross_entropies)
print(sum_cross)
