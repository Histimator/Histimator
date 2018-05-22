from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Credit to Keras implemetation :
# This is the default internal TF session used by Histimator.
# It can be set manually via `set_session(sess)`.
_SESSION = None


def tensor(x, name, dtype=np.float32):
    return np.asarray(x).astype(dtype)


def to_tensor(x, dtype=np.float32):
    return tf.convert_to_tensor(x, dtype=dtype)


def is_sparse(tensor):
    return isinstance(tensor, tf.SparseTensor)


def to_dense(tensor):
    if is_sparse(tensor):
        return tf.sparse_tensor_to_dense(tensor)
    else:
        return tensor


def variable(x, name=None, dtype=np.float32):
    return tf.variable(x, name=name, dtype=dtype)


def constant(x, name=None, dtype=np.float32):
    return tf.constant(x, name=name, dtype=dtype)


def sum(tensor):
    return tf.reduce_sum(tensor)


def prod(tensor):
    return tf.reduce_prod(tensor)


def xlogy(x, y):
    return tf.where(
        tf.equal(x, 0),
        tf.constant(0, dtype=np.float32),
        tf.prod(x, tf.log(y))
    )


class Normal(object):
    def __init__(self, mu=0, scale=0):
        self.mu = variable(mu)
        self.scale = variable(scale)
        self.pdf_ = tf.distributions.Normal(loc=mu, scale=scale)

    def log_prob(self, x):
        x = to_tensor(x)
        return self.pdf_.log_prob(x)

    def prob(self, x):
        x = to_tensor(x)
        return self.pdf_.pdf(x)


class cpoisson(object):
    def __init__(self, k=0, mu=1):
        self.k = variable(k)
        self.mu = variable(mu)

    def log_prob(self, k, mu):
        if k is None or mu is None:
            self.k = variable(k)
            self.mu = variable(mu)
        return tf.where(
            tf.greater(mu, 0),
            xlogy(k, mu) - tf.lgamma(k+1) - mu, 0)

    def pdf(self, k, mu):
        return tf.exp(self.logpdf(k, mu))
