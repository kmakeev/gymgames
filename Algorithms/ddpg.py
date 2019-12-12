import tensorflow as tf
import tensorflow_probability as tfp

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

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv1')

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv3')
        self.conv4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[7, 7], strides=1,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation='relu', use_bias=False, name='conv4')




        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(128, activation_fn)
        # self.build_visual = True

    def call(self, visual_input):
        features = visual_input / 255
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.flatten(features)
        features = self.fc(features)
        return features


class critic_q_all(ImageNet):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=visual_dim))

    @tf.function
    def call(self, visual_input):
        q = self.net(super().call(visual_input))
        return q


class critic_dueling(ImageNet):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.adv = mlp(hidden_units['adv'], output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=visual_dim))

    def call(self, visual_input):
        features = self.share(super().call(visual_input))
        v = self.v(features)
        adv = self.adv(features)
        return v, adv

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

class critic_q_one(ImageNet):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, visual_dim, action_dim, name, hidden_units):
        super().__init__(name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=visual_dim), tf.keras.Input(shape=action_dim))

    def call(self, visual_input, action):
        features = tf.concat((super().call(visual_input), action), axis=-1)
        q = self.net(features)
        return q


class DDPG(tf.keras.Model):
    def __init__(self,
                 visual_resolution,
                 a_counts,
                 max_episode,
                 base_dir,
                 gamma,
                 assign_interval=1000,
                 ployak=0.995,
                 discrete_tau=1.0,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 hidden_units={
                     'actor_continuous': [64, 64],
                     'actor_discrete': [64, 64],
                     'q': [64, 64]
                 }):
        super().__init__()
        self.visual_dim = visual_resolution
        self.a_counts = a_counts
        self.max_episode = max_episode
        self.gamma = gamma
        self.episode = 0  # episode of now
        self.IS_w = 1
        self.assign_interval = assign_interval
        self.discrete_tau = discrete_tau
        self.ployak = ployak

        self.actor_net = actor_discrete(self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'])
        self.actor_target_net = actor_discrete(self.visual_dim, self.a_counts, 'actor_target_net', hidden_units['actor_discrete'])
        self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        self.q_net = critic_q_one(self.visual_dim, self.a_counts, 'q_net', hidden_units['q'])
        self.q_target_net = critic_q_one(self.visual_dim, self.a_counts, 'q_target_net', hidden_units['q'])
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q_target_net.weights,
            self.actor_net.weights + self.q_net.weights
        )
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))


    def update_target_net_weights(self, tge, src, ployak=None):
        '''
        update weights of target neural network.
        '''
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(tge, src)])
        else:
            tf.group([t.assign(self.ployak * t + (1 - self.ployak) * s) for t, s in zip(tge, src)])

    def choose_action(self, visual_s):
        a = self._get_action(visual_s).numpy()
        # return sth.int2action_index(a, self.a_dim_or_list)
        return a

    @tf.function
    def _get_action(self, visual_s):
        logits = self.actor_net(visual_s)
        return tf.argmax(logits, axis=1)

    def learn(self, visual_s, a, r, visual_s_, done, **kwargs):
        self.episode = kwargs['episode']
        # for i in range(kwargs['step']):
        # if self.data.is_lg_batch_size:
        # s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
        # if self.use_priority:
        #    self.IS_w = self.data.get_IS_w()
        td_error, summaries = self.train(visual_s, a, r, visual_s_, done)

        summaries.update(dict([
            ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
            ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
        ]))
        # self.write_training_summaries(self.global_step, summaries)
        return summaries

    @tf.function(experimental_relax_shapes=True)
    def train(self, visual_s, a, r, visual_s_, done):
        # s, visual_s, a, r, s_, visual_s_, done = self.cast(visual_s, a, r, visual_s_, done)
        # with tf.device(self.device):
        a = tf.one_hot(a, self.a_counts, dtype=tf.float32)
        with tf.GradientTape() as tape:
            target_logits = self.actor_target_net(visual_s_)
            target_cate_dist = tfp.distributions.Categorical(target_logits)
            target_pi = target_cate_dist.sample()
            action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
            q = self.q_net(visual_s, a)
            q_target = self.q_target_net(visual_s_, action_target)
            dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
            td_error = q - dc_r
            q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * self.IS_w)
        q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
        self.optimizer_critic.apply_gradients(
            zip(q_grads, self.q_net.trainable_variables)
        )
        with tf.GradientTape() as tape:

            logits = self.actor_net(visual_s)
            logp_all = tf.nn.log_softmax(logits)
            gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
            _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
            _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
            _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
            pi = _pi_diff + _pi
            q_actor = self.q_net(visual_s, pi)
            actor_loss = -tf.reduce_mean(q_actor)
        actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
        self.optimizer_actor.apply_gradients(
            zip(actor_grads, self.actor_net.trainable_variables)
        )
        return td_error, dict([
            ['LOSS/loss', q_loss],
            # ['Statistics/q_max', tf.reduce_max(q_eval)],
            # ['Statistics/q_min', tf.reduce_min(q_eval)],
            # ['Statistics/q_mean', tf.reduce_mean(q_eval)]
        ])
