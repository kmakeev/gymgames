from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class MyModel(tf.keras.Model):

    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
        super(MyModel, self).__init__()
        """
        Args:
            n_actions: Integer, number of possible actions
            hidden: Integer, Number of filters in the final convolutional layer.
                    This is different from the DeepMind implementation
            learning_rate: Float, Learning rate for the Adam optimizer
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
        """
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length


        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv1')

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv3')
        self.conv4 =tf.keras.layers.Conv2D(filters=hidden, kernel_size=[7, 7], strides=1,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                           padding="valid", activation='relu', use_bias=False, name='conv4')

        # Splitting into value and advantage stream
        #self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.keras.layers.Flatten()
        self.advantagestream = tf.keras.layers.Flatten()
        self.advantage = tf.keras.layers.Dense(units=self.n_actions,
                                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                               name="advantage", activation='relu')
        self.value = tf.keras.layers.Dense(units=1,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                           name='value', activation='relu')

        # Combining value and advantage into Q-values as described above
        # self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        #self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        #self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # self.target_q = tf.compat.v2.keras.Input(shape=(), dtype=tf.float32, name='target_g')
        # Action that was performed
        # self.action = tf.compat.v2.keras.Input(shape=(), dtype=tf.int32, name='action')
        # self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        # self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)

        # Parameter updates
        # self.loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        # self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        #self.update = self.optimizer.minimize(self.loss)

    @tf.function
    def call(self, inputs):
        # Normalizing the input
        inputscaled = inputs/255
        conv1 = self.conv1(inputscaled)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        valuestream, advantagestream = tf.split(conv4, 2, 3)
        valuestream = self.valuestream(valuestream)
        advantagestream = self.advantagestream(advantagestream)
        advantage = self.advantage(advantagestream)
        value = self.value(valuestream)
        q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        return q_values

    @tf.function
    def best_action(self, inputs):
        q_values = self.call(inputs)
        best_action = tf.argmax(q_values, 1)
        return best_action

    """
    # @tf.function
    def update_loss(self, inputs, target_q, actions):
        q_values = self.call(inputs)
        Q = tf.reduce_sum(tf.multiply(q_values, tf.one_hot(actions, self.n_actions, dtype=tf.float32)), axis=1)
        loss = lambda: tf.reduce_mean(tf.losses.huber_loss(labels=target_q, predictions=Q))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        var_list_fn = lambda: self.trainable_weights

        update = self.optimizer.minimize(loss, var_list_fn)

        return (self.loss, update)
    """