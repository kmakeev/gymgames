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
                                               name="advantage")
        self.value = tf.keras.layers.Dense(units=1,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                           name='value')
        # self.lambda_layer = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
        # self.combine = tf.keras.layers.Add()


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
        # norm_advantage = self.lambda_layer(advantage)
        # combined = self.combine([value, norm_advantage])
        return q_values

    @tf.function
    def best_action(self, inputs):
        q_values = self.call(inputs)
        best_action = tf.argmax(q_values, 1)
        return best_action

    @tf.function
    def Q(self, inputs, actions):
        q_values = self.call(inputs)
        one_hot = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
        multiply = tf.multiply(q_values, one_hot)
        Q = tf.reduce_sum(multiply, axis=1)
        return Q
