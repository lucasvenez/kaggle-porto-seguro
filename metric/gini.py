import tensorflow as tf

class Gini(object):

    def __init__(self, session=tf.Session()):
        self.__session=session
        self.__actual = tf.placeholder(tf.float64, [None, None], name='gini_actual_value')
        self.__pred = tf.placeholder(tf.float64, [None, None], name='gini_estimated_value')
        self.__model = None
        self.__build()

    def __build(self):

        n = tf.shape(self.__actual)[1]

        indices = tf.reverse(tf.nn.top_k(self.__pred, k=n)[1], axis=[1])[0]

        a_s = tf.gather(tf.transpose(self.__actual), tf.transpose(indices))

        a_c = tf.cumsum(a_s)

        giniSum = tf.reduce_sum(a_c) / tf.reduce_sum(a_s)

        giniSum = tf.subtract(tf.cast(giniSum, tf.float32), tf.divide(tf.to_float(n + 1), tf.constant(2.)))

        self.__model = giniSum / tf.to_float(n)


    def calculate(self, a, p):
        return self.__session.run(self.__model, feed_dict = {self.__actual: a, self.__pred: p}) / \
               self.__session.run(self.__model, feed_dict={self.__actual: a, self.__pred: a})
