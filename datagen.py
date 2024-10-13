import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def problem0927(m: int, n: int, j_max: int) -> tuple[tf.Tensor, tf.Tensor]:
    '''
    Generate data
    :param m: vector dimension (input and output)
    :param n: number of points
    :param j_max: j is in [0, j_max]. There are j 1s and m-j 0s in input vector.
    :return: x: input vector; y: output vector.
    '''
    j = tf.random.uniform(minval=0, maxval=j_max + 1, shape=(n,), dtype=tf.dtypes.int32)
    x = []
    y = []
    for i in range(n):
        x.append(tf.random.shuffle(tf.concat(
            [tf.constant(1., shape=(j[i],), dtype=tf.dtypes.float16),
             tf.constant(0., shape=(m - j[i],), dtype=tf.dtypes.float16)], 0)))
        y.append(tf.constant(tf.cast(tf.range(0, m) < tf.constant(j[i], shape=(m,)), dtype=tf.dtypes.float32),
                             dtype=tf.dtypes.float32))
    x = tf.stack(x)
    y = tf.stack(y)
    return x, y


def hydrogen211(n, max_r) -> tuple[tf.Tensor, tf.Tensor]:
    '''
    the (2,1,1) state of hydrogen atom
    '''
    def psi(x: tf.Tensor):
        r = tf.expand_dims(x[:,0], axis=-1)
        theta = tf.expand_dims(x[:,1], axis=-1)
        phi = tf.expand_dims(x[:,2], axis=-1)
        return -r * tf.math.exp(-r / 2) * tf.math.sin(theta) * tf.math.cos(phi) / (8 * np.sqrt(np.pi))
    x = tf.transpose(tf.stack([
        max_r * tf.random.uniform((n,), dtype=tf.dtypes.float32, seed=0) ** (1. / 3),
        tf.random.uniform((n,), 0., 2 * np.pi, dtype=tf.dtypes.float32, seed=1),
        - tf.math.acos(tf.random.uniform((n,), -1., 1., dtype=tf.dtypes.float32, seed=2))
    ]))
    psi_ = psi(x)
    return x, psi_

def trig(n, max_x) -> tuple[tf.Tensor, tf.Tensor]:
    '''
    a sine function
    '''
    x = tf.expand_dims(tf.random.uniform((n,), 0., max_x, dtype=tf.dtypes.float32), axis=-1)
    y = tf.math.sin(x)
    return x, y
