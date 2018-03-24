import tensorflow as tf
import numpy as np
import metric

from time import gmtime, strftime

class MLP(object):

    def __init__(self, n_inputs, hidden_neurons, n_outputs=1, activation_function=tf.nn.relu, session=tf.Session()):

        self.__id = strftime("%Y%m%d%H%M%S", gmtime()) + '-mlp-porto-seguro'

        self.__session = session

        self.__input = tf.placeholder(tf.float32, [None, None], name='input')
        self.__output = tf.placeholder(tf.float32, [None, n_outputs], 'output')
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.__weights = []
        self.__biases  = []

        self.__model = None

        self.__gini = metric.Gini()

        self.__build(n_inputs, hidden_neurons, n_outputs, activation_function)

    def __build(self, n_inputs, hidden_neurons, n_outputs, activation_function=tf.nn.relu, learning_rate=0.005):

        self.__model = self.__input

        last_size = n_inputs

        for i in range(len(hidden_neurons)):
            self.__weights.append(tf.Variable(tf.truncated_normal([last_size, hidden_neurons[i]], stddev=.1)))
            self.__biases.append(tf.Variable(tf.truncated_normal([hidden_neurons[i]], stddev=.1)))

            last_size = hidden_neurons[i]

            self.__model = tf.nn.dropout(self.__model, self.__keep_prob)
            self.__model = activation_function(tf.add(tf.matmul(self.__model, self.__weights[i]), self.__biases[i]))

        self.__model = tf.nn.dropout(self.__model, self.__keep_prob)
        self.__weights.append(tf.Variable(tf.truncated_normal([last_size, n_outputs], stddev=.1)))
        self.__biases.append(tf.Variable(tf.truncated_normal([n_outputs], stddev=.1)))

        self.__model = tf.add(tf.matmul(self.__model, self.__weights[-1]), self.__biases[-1])

        self.__loss = tf.nn.l2_loss(tf.abs(self.__output - self.__model))
        #self.__loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.__output, logits=self.__model))
        #self.__loss = tf.reduce_mean(tf.sqrt(tf.square(self.__model - self.__output)))
        #self.__loss = tf.reduce_mean(tf.square(self.__output - self.__model))

        self.__optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength=.2)
        self.__optimizer = self.__optimizer.minimize(self.__loss)

    def optimize(self, x, y, steps=10000, batch_size=None, shuffle=True, x_test=None, y_test=None):

        if batch_size is None:
            batch_size = x.shape[0]

        if x_test is None:
            x_test = x

        if y_test is None:
            y_test = y

        self.__session.run(tf.global_variables_initializer())

        n_rows = x.shape[0]

        index = np.array(list(range(n_rows)), dtype=np.int)

        for step in range(steps):

            current_block = 0

            while (current_block < n_rows):

                if shuffle:
                    np.random.shuffle(index)

                batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                loss, _ = self.__session.run([self.__loss, self.__optimizer], feed_dict={self.__input: x[index[batch], :], self.__output: y[index[batch], :], self.__keep_prob: .5})

                y_hat = self.__session.run(self.__model, feed_dict={self.__input: x_test, self.__keep_prob: 1.})

                current_block += batch_size

            gini_value = None

            if (step + 1) % 10  == 0 or (step + 1) == steps:
                gini_value = self.__gini.calculate(y_test.T, np.array(y_hat).T)
                print('Step {} of {}. Loss: {}. Gini: {}'.format(step + 1, steps, loss, gini_value))

            else:
                print('Step {} of {}. Loss: {}.'.format(step + 1, steps, loss))

            if (step + 1) % 500 == 0 or (step + 1) == steps:
                gini_value = self.__gini.calculate(y_test.T, np.array(y_hat).T)
                self.export('-' + str(int(gini_value*1000)) + '-' + str(step + 1))


    def predict(self, x):
        return self.__session.run(self.__model, feed_dict={self.__input: x, self.__keep_prob: 1.})

    def export(self, sufix):
        saver = tf.train.Saver()
        saver.save(self.__session, './output/model/{}'.format(self.__id + sufix))
