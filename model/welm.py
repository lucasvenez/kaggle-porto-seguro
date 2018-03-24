import tensorflow as tf
import numpy as np

# CHECK : Constants
omega = 1.

class WELM(object):
  def __init__(self, batch_size, input_len, hidden_num, output_len, sess=tf.Session()):
    '''
    Args:
      sess : TensorFlow session.
      batch_size : The batch size (N)
      input_len : The length of input. (L)
      hidden_num : The number of hidden node. (K)
      output_len : The length of output. (O)
    '''

    self._sess = sess
    self._batch_size = batch_size
    self._input_len = input_len
    self._hidden_num = hidden_num
    self._output_len = output_len

    # for train
    self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
    self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

    # for test
    self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
    self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

    self._W = tf.Variable(
      tf.random_normal([self._input_len, self._hidden_num], stddev=.1, seed=49),
      trainable=False, dtype=tf.float32)

    self._b = tf.Variable(
      tf.random_normal([self._hidden_num], stddev=.1, seed=49),
      trainable=False, dtype=tf.float32)

    self._beta = tf.Variable(
      tf.zeros([self._hidden_num, self._output_len]),
      trainable=False, dtype=tf.float32)

    self._WEIGHTS = tf.placeholder(tf.float32, [self._batch_size])

    self._var_list = [self._W, self._b, self._beta]

    self.H0 = tf.matmul(self._x0, self._W) + self._b

    self.H0_T = tf.transpose(self.H0)

    self.H1 = tf.matmul(self._x1, self._W) + self._b
    self.H1_T = tf.transpose(self.H1)

    if self._batch_size < self._hidden_num:
      identity = tf.constant(np.ones(self._batch_size)/omega, dtype=tf.float32)
      self._beta_s = tf.matmul(
                        tf.matmul(
                           self.H0_T,
                           tf.matrix_inverse(
                              tf.add(
                                 identity,
                                 tf.matmul(
                                    tf.multiply(
                                       self.H0,
                                       tf.expand_dims(self._WEIGHTS,-1)
                                    ),
                                    self.H0_T
                                 )
                              )
                           )
                        ),
                        tf.multiply(
                           tf.expand_dims(self._WEIGHTS, -1),
                           self._t0
                        )
                     )

    else:
      identity = tf.constant(np.ones(self._hidden_num)/omega, dtype=tf.float32)
      self._beta_s = tf.matmul(
                        tf.matrix_inverse(
                           tf.add(
                              identity,
                              tf.matmul(
                                 tf.multiply(
                                    self.H0_T,
                                    tf.transpose(tf.expand_dims(self._WEIGHTS,-1))
                                 ),
                                 self.H0
                              )
                           )
                        ),
                        tf.matmul(
                           tf.multiply(
                              self.H0_T,
                              tf.transpose(tf.expand_dims(self._WEIGHTS, -1))),
                           self._t0
                        )
                     )

    self._assign_beta = self._beta.assign(self._beta_s)
    self._fx0 = tf.matmul(self.H0, self._beta)
    self._fx1 = tf.matmul(self.H1, self._beta)

    self._init = False
    self._feed = False

  def feed(self, x, t, w= np.array([1., 0.618])):
    '''
    Args :
      x : input array (N x L)
      t : label array (N x O)
    '''

    if not self._init : self.init()

    u = np.unique(t)

    count = np.array([(t == i).sum() for i in u])

    weights = (w[t] /(count[t]))[:,0]

    self._sess.run(self._assign_beta, {self._x0:x, self._t0:t, self._WEIGHTS: weights})

    self._feed = True

  def init(self):
    self._sess.run(tf.variables_initializer(self._var_list))
    self._init = True

  def test(self, x, t=None):
    if not self._feed : exit("Not feed-forward trained")
    if t is not None :
      print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1:x, self._t1:t})))
    else :
      return self._sess.run(self._fx1, {self._x1:x})
