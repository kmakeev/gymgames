from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# from utils import sth
# from tf2_utils import get_TensorSpecs, gaussian_clip_reparam_sample, gaussian_likelihood, gaussian_entropy

activation_fn = tf.keras.activations.tanh


class mlp(tf.keras.Sequential):
    def __init__(self, hidden_units, act_fn=activation_fn, output_shape=1, out_activation=None, out_layer=True):
        """
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(tf.keras.layers.Dense(u, act_fn))
        if out_layer:
            self.add(tf.keras.layers.Dense(output_shape, out_activation))


class ImageNet(tf.keras.Model):
    '''
    Processing image input observation information.
    If there has multiple cameras, Conv3D will be used, otherwise Conv2D will be used. The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, name):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4, padding='valid',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            activation=activation_fn, use_bias=False, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2, padding='valid',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            activation=activation_fn, use_bias=False, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='valid',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            activation=activation_fn, use_bias=False, name='conv3')
        self.conv4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[7, 7], strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            activation=activation_fn, use_bias=False, name='conv4')

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(128, activation_fn)

    def call(self, visual_input):
        features = visual_input / 255
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.flatten(features)
        features = self.fc(features)
        return features


class actor_discrete(ImageNet):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name)
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=visual_dim))

    def call(self, visual_input):
        logits = self.logits(super().call(visual_input))
        return logits


class critic_v(ImageNet):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, visual_dim, name, hidden_units):
        super().__init__(name=name)
        self.net = mlp(hidden_units, act_fn=activation_fn, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=visual_dim))

    def call(self, visual_input):
        v = self.net(super().call(visual_input))
        return v


class A2C(tf.keras.Model):
    def __init__(self,
                 visual_resolution,
                 a_counts,
                 max_episode,
                 base_dir,
                 gamma=0.99,
                 assign_interval=1000,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 hidden_units={
                     'actor_continuous': [128, 128],
                     'actor_discrete': [128, 128],
                     'critic': [128, 128]
                 },
                 logger2file=False,
                 out_graph=False):
        super().__init__()

        self.visual_dim = visual_resolution
        self.gamma = gamma
        self.max_episode = max_episode
        self.beta = beta
        self.a_counts = a_counts
        self.episode = 0  # episode of now
        self.IS_w = 1  # the weights of NN variables by using Importance sampling.

        self.actor_net = actor_discrete(self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'])
        self.critic_net = critic_v(self.visual_dim, 'critic_net', hidden_units['critic'])
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(lr*5, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))

    def choose_action(self, visual_s):
        a = self._get_action(visual_s).numpy()
        return a

    @tf.function
    def _get_action(self, visual_input):
        logits = self.actor_net(visual_input)
        norm_dist = tfp.distributions.Categorical(logits)
        sample_op = norm_dist.sample()
        return sample_op

    def learn(self, visual_s, a, r, visual_s_, done, **kwargs):
        self.episode = kwargs['episode']
        actor_loss, critic_loss, entropy = self.train(visual_s, a, r)
        return dict([['LOSS/loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],])

    @tf.function(experimental_relax_shapes=True)
    def train(self, visual_s, a, dc_r):
        a = tf.one_hot(a, self.a_counts, dtype=tf.float32)
        with tf.GradientTape() as tape:
            v = self.critic_net(visual_s)
            td_error = dc_r - v
            critic_loss = tf.reduce_mean(tf.square(td_error))
        critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
        self.optimizer_critic.apply_gradients(
            zip(critic_grads, self.critic_net.trainable_variables)
        )
        with tf.GradientTape() as tape:
            logits = self.actor_net(visual_s)
            logp_all = tf.nn.log_softmax(logits)
            log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
            v = self.critic_net(visual_s)
            advantage = tf.stop_gradient(dc_r - v)
            actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
        return actor_loss, critic_loss, entropy
