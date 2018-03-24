import tensorflow as tf
import numpy as np

class DualAutoencoder(object):

    def __init__(self, n_inputs, n_hidden_neurons=[8, 8, 8, 8], session=tf.Session()):
        self.__session = session
        self.__model = None

        self.__losses = [None, None, None, None]
        self.__optimizers = [None, None, None, None]

        self.__build(n_inputs, n_hidden_neurons)


    def __build(self, n_inputs, n_hidden_neurons):

        self.__weights_encoders_sigmoid  = [tf.Variable(tf.random_normal([n_inputs, n_hidden_neurons[0]], stddev=.1)),
                                          tf.Variable(tf.random_normal([n_hidden_neurons[0], n_hidden_neurons[1]], stddev=.1))]

        self.__biases_encoders_sigmoid  = [tf.Variable(tf.random_normal([n_hidden_neurons[0]], stddev=.1)),
                                          tf.Variable(tf.random_normal([n_hidden_neurons[1]], stddev=.1))]

        self.__weights_decoders_sigmoid = [tf.Variable(tf.random_normal([n_hidden_neurons[0], n_inputs], stddev=.1)),
                                          tf.Variable(tf.random_normal([n_hidden_neurons[1], n_inputs], stddev=.1))]

        self.__biases_decoders_sigmoid  = [tf.Variable(tf.random_normal([n_inputs], stddev=.1)),
                                          tf.Variable(tf.random_normal([n_inputs], stddev=.1))]

        self.__weights_encoders_tanh = [tf.Variable(tf.random_normal([n_inputs, n_hidden_neurons[2]], stddev=.1)),
                                       tf.Variable(tf.random_normal([n_hidden_neurons[2], n_hidden_neurons[3]], stddev=.1))]

        self.__biases_encoders_tanh  = [tf.Variable(tf.random_normal([n_hidden_neurons[2]], stddev=.1)),
                                        tf.Variable(tf.random_normal([n_hidden_neurons[3]], stddev=.1))]

        self.__weights_decoders_tanh = [tf.Variable(tf.random_normal([n_hidden_neurons[2], n_inputs], stddev=.1)),
                                       tf.Variable(tf.random_normal([n_hidden_neurons[3], n_inputs], stddev=.1))]

        self.__biases_decoders_tanh = [tf.Variable(tf.random_normal([n_inputs], stddev=.1)),
                                       tf.Variable(tf.random_normal([n_inputs], stddev=.1))]

        self.__input = tf.placeholder(tf.float32, [None, n_inputs])

        encoder_sigmoid = [None, None]
        decoder_sigmoid = [None, None]

        encoder_sigmoid[0] = self.__encoder(self.__input,       self.__weights_encoders_sigmoid[0], self.__biases_encoders_sigmoid[0], tf.sigmoid)
        decoder_sigmoid[0] = self.__decoder(encoder_sigmoid[0], self.__weights_decoders_sigmoid[0], self.__biases_decoders_sigmoid[0], tf.sigmoid)

        encoder_sigmoid[1] = self.__encoder(encoder_sigmoid[0], self.__weights_encoders_sigmoid[1], self.__biases_encoders_sigmoid[1], tf.identity)
        decoder_sigmoid[1] = self.__decoder(encoder_sigmoid[1], self.__weights_decoders_sigmoid[1], self.__biases_decoders_sigmoid[1], tf.identity)

        encoder_tanh = [None, None]
        decoder_tanh = [None, None]

        encoder_tanh[0] = self.__encoder(self.__input,    self.__weights_encoders_tanh[0], self.__biases_encoders_tanh[0], tf.tanh)
        decoder_tanh[0] = self.__decoder(encoder_tanh[0], self.__weights_decoders_tanh[0], self.__biases_decoders_tanh[0], tf.tanh)

        encoder_tanh[1] = self.__encoder(encoder_tanh[0], self.__weights_encoders_tanh[1], self.__biases_encoders_tanh[1], tf.identity)
        decoder_tanh[1] = self.__decoder(encoder_tanh[1], self.__weights_decoders_tanh[1], self.__biases_decoders_tanh[1], tf.identity)

        self.encoders = encoder_sigmoid + encoder_tanh
        self.decoders = decoder_sigmoid + decoder_tanh

        for i in range(4):
            self.__losses[i] = tf.reduce_mean(tf.square(self.decoders[i] - self.__input))
            self.__optimizers[i] = tf.train.FtrlOptimizer(learning_rate=.2, l2_regularization_strength=.2).minimize(self.__losses[i])


    def __encoder(self, x, weights, biases, activation_function):
        return self.__dense(x, weights, biases, activation_function)


    def __decoder(self, x, weights, biases, activation_function):
        return self.__dense(x, weights, biases, activation_function)


    def __dense(self, x, weights, biases, activation_function):
        return activation_function(tf.add(tf.matmul(x, weights), biases))


    def optimize(self, x, shuffle=True, steps=1000, batch_size=100):

        self.__session.run(tf.global_variables_initializer())

        n_rows = x.shape[0]

        index = np.array(list(range(n_rows)), dtype=np.int)

        for step in range(steps):

            current_block = 0

            while (current_block < n_rows):

                if shuffle:
                    np.random.shuffle(index)

                batch = list(range(current_block, (min(current_block + batch_size, n_rows))))

                l1, l2, l3, l4, _, _, _, _ = self.__session.run(self.__losses + self.__optimizers,
                                                                feed_dict={self.__input: x[index[batch],:]})

                current_block += batch_size

                if (step + 1) % 2 == 0:
                    print('Step {0: <5} of {1: <5}. Losses: {2: <6}, {3: <6}, {4: <6}, {5: <6}'.format(step + 1, steps, str(round(l1, 4)), str(round(l2, 4)), str(round(l3, 4)), str(round(l4, 4))))


    def predict(self, x):

        e1, e2, e3, e4 = self.__session.run(self.encoders, feed_dict={self.__input: x})

        return np.concatenate((e2, e4), axis=1)


    def export(self):
        pass