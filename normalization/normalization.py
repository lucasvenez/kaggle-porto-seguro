import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer


class Normalization(object):


    def fit(self, x):
        pass


    def fit_and_normalize(self, x):
        self.fit(x)
        return self.normalize(x)


    def normalize(self, x):
        pass


class ZScore(Normalization):

    def __init__(self, session=tf.Session()):
        self.session = session
        self.input = tf.placeholder(tf.float64, [None, None])
        self.tf_mean, tf_var = tf.nn.moments(self.input, axes=[0])
        self.tf_std = tf.sqrt(tf_var)


    def fit(self, x):

        self.tf_mean_var = tf.Variable(tf.zeros([x.shape[1]], dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.tf_std_var = tf.Variable(tf.zeros([x.shape[1]], dtype=tf.float64), trainable=False, dtype=tf.float64)

        tf_mean_assign = tf.assign(self.tf_mean_var, self.tf_mean)
        tf_std_assign = tf.assign(self.tf_std_var, self.tf_std)

        self.session.run([tf_mean_assign, tf_std_assign], feed_dict={self.input: x})
        self.tf_zscore = (self.input - self.tf_mean_var) / self.tf_std_var


    def normalize(self, x):
        return self.session.run(self.tf_zscore, feed_dict={self.input: x})


    def fit_and_normalize(self, x):
        self.fit(x)
        return self.normalize(x)


class MinMax(Normalization):


    def __init__(self, session=tf.Session()):
        self.session = session
        self.input = tf.placeholder(tf.float64, [None, None])
        self.tf_max = tf.reduce_max(self.input, axis=0)
        self.tf_min = tf.reduce_min(self.input, axis=0)


    def fit(self, x):

        self.tf_max_var = tf.Variable(tf.zeros([x.shape[1]], dtype=tf.float64), trainable=False, dtype=tf.float64)
        self.tf_min_var = tf.Variable(tf.zeros([x.shape[1]], dtype=tf.float64), trainable=False, dtype=tf.float64)

        tf_max_assign = tf.assign(self.tf_max_var, self.tf_max)
        tf_min_assign = tf.assign(self.tf_min_var, self.tf_min)

        self.session.run([tf_max_assign, tf_min_assign], feed_dict={self.input: x})

        self.tf_min_max = (self.input - self.tf_min_var) / (self.tf_max_var - self.tf_min_var)


    def normalize(self, x):
        return self.session.run(self.tf_min_max, feed_dict={self.input: x})


class BoxCox(Normalization):

    def __init__(self, session=tf.Session()):

        self.__ALPHA = tf.reshape(tf.constant([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], dtype=tf.float64), [-1, 1], name="alpha")

        self.__input = tf.placeholder(shape=[None], dtype=tf.float64, name="boxcox_input")

        self.__dependent_variable = tf.placeholder(shape=[None], dtype=tf.float64, name="boxcox_dependent_var")

        self.__session = session

        self.__boxcox_params = []


    def fit(self, x, y):

        if len(x.shape) == 1:
            x = np.array([x]).T

        self.__n = len(y)

        op_with_inf = tf.divide(tf.subtract(tf.pow(self.__input, self.__ALPHA), tf.constant(1., dtype=tf.float64)), self.__ALPHA)

        op = tf.where(tf.is_inf(op_with_inf), tf.zeros_like(op_with_inf), op_with_inf)

        k_1 = self.__n * tf.matmul(op, tf.reshape(self.__dependent_variable, [-1, 1]))

        k_2 = tf.reduce_sum(op, axis=1, keep_dims= True)

        k_3 = tf.reshape(tf.reduce_sum(tf.ones([9, 1], dtype=tf.float64) * tf.reduce_sum(self.__dependent_variable), axis=1), [-1, 1])

        k_4 = self.__n * tf.reduce_sum(tf.square(op), axis=1, keep_dims= True) - tf.square(k_2)

        k_5 = self.__n * tf.reshape(tf.reduce_sum(tf.ones([9, 1], dtype=tf.float64) *
              tf.reduce_sum(tf.square(self.__dependent_variable)), axis=1), [-1, 1]) - tf.square(k_3)

        self.__operation = tf.argmax((k_1 - k_2 * k_3) / tf.sqrt(k_4 * k_5))

        for i in range(x.shape[1]):

            index = (self.__session.run(self.__operation, feed_dict={self.__input: x[:,i], self.__dependent_variable: y}))

            self.__boxcox_params.append(self.__session.run(self.__ALPHA)[index,0][0])


    def normalize(self, x):

        if len(x.shape) == 1:
            x = np.array([x]).T

        if (x.shape[1] != len(self.__boxcox_params)):
            raise ValueError('normalize input should have the same number of columns of the fit input')

        result = np.zeros([x.shape[0], x.shape[1]])

        for i in range(x.shape[1]):

            boxcox = self.__boxcox_params[i]

            for j in range(len(x[:, i])):
                if x[j,i] == 0 and boxcox < 0:
                    result[j, i] = 0.
                else:
                    result[j, i] = np.power(x[j,i], boxcox) if boxcox != 0 else np.log(x[j, i])

        return result


    def fit_and_normalize(self, x, y):
        self.fit(x, y)
        return self.normalize(x)


    def get_params(self):
        return self.__boxcox_params

class FeatureBinarizatorAndScaler(Normalization):
    """ This class needed for scaling and binarization features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])


    def normalize(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((
                len(data[feature]), 1))), axis=1)
        print(binarizedAndScaledFeatures.shape)
        return binarizedAndScaledFeatures