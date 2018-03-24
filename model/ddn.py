from time import gmtime, strftime

import tensorflow as tf
import numpy as np
import metric

class DualDeepNetwork(object):

    def __init__(self, n_inputs, n_hidden_neurons=[8, 8], n_outputs=1, session=tf.Session()):

        self.__session = session

        self.__id = strftime("%Y%m%d%H%M%S", gmtime()) + '-ddn-porto-seguro'

        self.__input = tf.placeholder(tf.float32, [None, n_inputs], name='input')
        self.__output = tf.placeholder(tf.float32, [None, n_outputs], name='expected_output')
        self.__keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.__input_layers = []

        self.__hidden_weights = []
        self.__hidden_biases = []
        self.__hidden_layers = []

        self.__output_weights = []
        self.__output_biases = []
        self.__output_layers = []

        self.__losses = []
        self.__optimizers = []

        self.__gini = metric.Gini()

        self.__build(n_inputs, n_hidden_neurons, n_outputs)


    def __build(self, n_inputs, n_hidden_neurons, n_outputs, learning_rate=.005, l2_regularization=.5, activation_function=tf.nn.relu):

        last_layer_size = n_inputs

        self.__input_layers.append(self.__input)

        for i in range(len(n_hidden_neurons)):

            self.__hidden_weights.append(tf.Variable(tf.random_normal([last_layer_size, n_hidden_neurons[i]], stddev=.1)))
            self.__hidden_biases.append(tf.Variable(tf.random_normal([n_hidden_neurons[i]], stddev=.1)))

            self.__hidden_layers.append(activation_function(tf.add(tf.matmul(tf.nn.dropout(self.__input_layers[-1], self.__keep_prob), self.__hidden_weights[-1]), self.__hidden_biases[-1])))

            self.__output_weights.append(tf.Variable(tf.random_normal([n_hidden_neurons[i], n_outputs], stddev=.1)))
            self.__output_biases.append(tf.Variable(tf.random_normal([n_outputs], stddev=.1)))

            self.__output_layers.append(tf.add(tf.matmul(tf.nn.dropout(self.__hidden_layers[-1], keep_prob=self.__keep_prob), self.__output_weights[-1]), self.__output_biases[-1], name='output_var_' + str(i)))

            self.__input_layers.append(self.__hidden_layers[-1])

            last_layer_size = n_hidden_neurons[i]

            self.__losses.append(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.__output, logits=self.__output_layers[i])))
            self.__optimizers.append(tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength=l2_regularization).minimize(self.__losses[i]))

    def optimize(self, x, y, shuffle=True, steps=1000, batch_size=None, keep_prob=.5, x_test=None, y_test=None, verbose_step=10):

        if x_test is None or y_test is None:
            x_test = x
            y_test = y

        if batch_size is None:
            batch_size = x.shape[0]

        self.__session.run(tf.global_variables_initializer())

        n_rows = x.shape[0]

        index = np.array(list(range(n_rows)), dtype=np.int)

        for step in range(steps):

            current_block = 0

            while (current_block < n_rows):

                if shuffle:
                    np.random.shuffle(index)

                batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                losses = self.__session.run(self.__losses + self.__optimizers, feed_dict={self.__input: x[index[batch],:], self.__output: y[index[batch],:], self.__keep_prob: keep_prob})

                current_block += batch_size

            if (step + 1) % verbose_step == 0 or step + 1 == steps:

                print('===========================================================')
                print('Step {0} of {1}'.format(step + 1, steps))
                print('===========================================================')

                for i in range(len(self.__losses)):
                    print('* Losses {}: {}'.format(i, losses[i]))

                y_hat = self.__session.run(self.__output_layers[-1], feed_dict={self.__input: x_test, self.__keep_prob: 1.})

                gini_value = self.__gini.calculate(y_test.T, np.array(y_hat).T)

                print('* Gini test: {}'.format(gini_value))

                print('===========================================================\n')

            if (step + 1) % 500 == 0 or (step + 1) % steps == 0:
                self.export('-' + str(int(gini_value*1000)) + '-' + str(step + 1))


    def predict(self, x):
        return self.__session.run(self.__output_layers, feed_dict={self.__input: x, self.__keep_prob: 1.})


    def export(self, sufix):
        saver = tf.train.Saver()
        saver.save(self.__session, './output/model/{}'.format(self.__id + sufix))


    def restore(self, id):
        saver = tf.train.import_meta_graph('./output/model/{}.meta'.format(id))
        saver.restore(self.session, "./output/model/{}".format(id))
