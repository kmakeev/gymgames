from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import sth
from tf2_utils import get_TensorSpecs, gaussian_clip_reparam_sample, gaussian_likelihood, gaussian_entropy


activation_fn = tf.keras.activations.tanh


class mlp(tf.keras.Sequential):
    def __init__(self, act_fn=activation_fn, output_shape=1, out_activation=None):
        """
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        self.add(tf.keras.layers.Dense(128, act_fn))
        self.add(tf.keras.layers.Dense(128, act_fn))
        self.add(tf.keras.layers.Dense(output_shape, out_activation))


class ImageNet(tf.keras.Model):
    '''
    Processing image input observation information.
    If there has multiple cameras, Conv3D will be used, otherwise Conv2D will be used. The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, name, visual_dim=[]):
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
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[7, 7], strides=1, padding='valid',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            activation=activation_fn, use_bias=False, name='conv4')

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(128, activation_fn)

    def call(self, vector_input, visual_input):
        features = self.conv1(visual_input)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.flatten(features)
        features = self.fc(features)
        vector_input = tf.concat((features, vector_input), axis=-1)
        return vector_input


class actor_discrete(ImageNet):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.logits = mlp(act_fn=activation_fn, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        logits = self.logits(super().call(vector_input, visual_input))
        return logits


class critic_v(ImageNet):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, visual_dim, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(act_fn=activation_fn, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        v = self.net(super().call(vector_input, visual_input))
        return v


class A2C(tf.keras.Model):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 max_episode=50000,
                 base_dir=None,
                 batch_size=128,
                 epoch=5,
                 beta=1.0e-3,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 logger2file=False,
                 out_graph=False):
        super().__init__()
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        if visual_sources == 1:
            self.visual_dim = visual_resolution
        elif visual_sources > 1:
            self.visual_dim = [visual_sources, *visual_resolution]
        else:
            self.visual_dim = [0]
        self.a_dim_or_list = a_dim_or_list
        self.gamma = gamma
        self.max_episode = max_episode
        self.batch_size = batch_size
        # self.init_data_memory()
        self.beta = beta
        self.epoch = epoch

        self.a_counts = int(np.array(a_dim_or_list).prod())
        self.episode = 0  # episode of now
        self.IS_w = 1  # the weights of NN variables by using Importance sampling.

        self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1])


        self.actor_net = actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'])
        self.critic_net = critic_v(self.s_dim, self.visual_dim, 'critic_net', hidden_units['critic'])
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))

    def choose_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu = self.actor_net(vector_input, visual_input)
                sample_op, _ = gaussian_clip_reparam_sample(mu, self.log_std)
            else:
                logits = self.actor_net(vector_input, visual_input)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.on_store(s, visual_s, a, r, s_, visual_s_, done)

    def calculate_statistics(self):
        s, visual_s = self.data.s_.values[-1], self.data.visual_s_.values[-1]
        init_value = np.squeeze(self.critic_net(s, visual_s))
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index + self.batch_size]
        s = np.vstack(i_data.s.values)
        visual_s = np.vstack(i_data.visual_s.values)
        a = np.vstack(i_data.a.values)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1)
        return s, visual_s, a, dc_r

    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.recorder.writer.set_as_default()
        self.calculate_statistics()
        for _ in range(self.epoch):
            for index in range(0, self.data.shape[0], self.batch_size):
                s, visual_s, a, dc_r = [tf.convert_to_tensor(i) for i in self.get_sample_data(index)]
                actor_loss, critic_loss, entropy = self.train.get_concrete_function(
                    *self.TensorSpecs)(s, visual_s, a, dc_r)
        self.global_step.assign_add(1)
        tf.summary.experimental.set_step(self.episode)
        tf.summary.scalar('LOSS/entropy', entropy)
        tf.summary.scalar('LOSS/actor_loss', actor_loss)
        tf.summary.scalar('LOSS/critic_loss', critic_loss)
        tf.summary.scalar('LEARNING_RATE/actor_lr', self.actor_lr(self.episode))
        tf.summary.scalar('LEARNING_RATE/critic_lr', self.critic_lr(self.episode))
        self.recorder.writer.flush()
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                v = self.critic_net(s, visual_s)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error))
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu = self.actor_net(s, visual_s)
                    log_act_prob = gaussian_likelihood(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(s, visual_s)
                advantage = tf.stop_gradient(dc_r - v)
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            if self.action_type == 'continuous':
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables + [self.log_std])
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables + [self.log_std])
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables)
                )
            return actor_loss, critic_loss, entropy

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, dc_r):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.action_type == 'continuous':
                    mu = self.actor_net(s, visual_s)
                    log_act_prob = gaussian_likelihood(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(s, visual_s)
                advantage = tf.stop_gradient(dc_r - v)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error))
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            if self.action_type == 'continuous':
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables + [self.log_std])
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables + [self.log_std])
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables)
                )
            return actor_loss, critic_loss, entropy
