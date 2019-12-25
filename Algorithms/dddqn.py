import tensorflow as tf

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


class DDDQN(tf.keras.Model):
    def __init__(self,
                 visual_resolution,
                 a_counts,
                 max_episode,
                 base_dir,
                 gamma,
                 assign_interval=1000,
                 lr=5.0e-4,
                 hidden_units={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 }):
        super().__init__()
        self.visual_dim = visual_resolution
        self.a_counts = a_counts
        self.max_episode = max_episode
        self.gamma = gamma
        self.episode = 0  # episode of now
        self.IS_w = 1
        self.assign_interval = assign_interval

        # self.q_net = critic_q_all(self.visual_dim, self.a_counts, 'q_net', hidden_units)
        # self.q_target_net = critic_q_all(self.visual_dim, self.a_counts, 'q_target_net', hidden_units)

        self.dueling_net = critic_dueling(self.visual_dim, self.a_counts, 'dueling_net', hidden_units)
        self.dueling_target_net = critic_dueling(self.visual_dim, self.a_counts, 'dueling_target_net',
                                                    hidden_units)
        self.update_target_net_weights()
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))

    def update_target_net_weights(self, ployak=None):
        '''
        update weights of target neural network.
        '''
        tge = self.dueling_target_net.weights
        src = self.dueling_net.weights
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
        _, advs = self.dueling_net(visual_s)
        return tf.argmax(advs, axis=1)

    def learn(self, visual_s, a, r, visual_s_, done, **kwargs):
        self.episode = kwargs['episode']
        # for i in range(kwargs['step']):
        # if self.data.is_lg_batch_size:
        # s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
        # if self.use_priority:
        #    self.IS_w = self.data.get_IS_w()
        td_error, summaries = self.train(visual_s, a, r, visual_s_, done)

        summaries.update(dict([['LEARNING_RATE/lr', self.lr(self.episode)]]))
        # self.write_training_summaries(self.global_step, summaries)
        return summaries

    @tf.function(experimental_relax_shapes=True)
    def train(self, visual_s, a, r, visual_s_, done):
        # s, visual_s, a, r, s_, visual_s_, done = self.cast(visual_s, a, r, visual_s_, done)
        # with tf.device(self.device):
        with tf.GradientTape() as tape:
            v, adv = self.dueling_net(visual_s)
            average_adv = tf.reduce_mean(adv, axis=1, keepdims=True)
            v_next, adv_next = self.dueling_net(visual_s_)
            next_max_action = tf.argmax(adv_next, axis=1, name='next_action_int')
            next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.a_counts, 1., 0., dtype=tf.float32)
            # next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)
            v_next_target, adv_next_target = self.dueling_target_net(visual_s_)
            average_a_target_next = tf.reduce_mean(adv_next_target, axis=1, keepdims=True)
            q_eval = tf.reduce_sum(tf.multiply(v + adv - average_adv, tf.one_hot(a, self.a_counts, dtype=tf.float32)), axis=1, keepdims=True)
            q_target_next_max = tf.reduce_sum(
                tf.multiply(v_next_target + adv_next_target - average_a_target_next, next_max_action_one_hot),
                axis=1, keepdims=True)
            q_target = tf.stop_gradient(r + self.gamma * (1 - done) * q_target_next_max)
            td_error = q_eval - q_target
            q_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
        grads = tape.gradient(q_loss, self.dueling_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dueling_net.trainable_variables)
        )
        # self.global_step.assign_add(1)
        return td_error, dict([
            ['LOSS/loss', q_loss],
            # ['Statistics/q_max', tf.reduce_max(q_eval)],
            # ['Statistics/q_min', tf.reduce_min(q_eval)],
            # ['Statistics/q_mean', tf.reduce_mean(q_eval)]
        ])
