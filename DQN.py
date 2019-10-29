from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import random


class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame = tf.Variable(tf.ones(shape=[210, 160, 3]))
        self.processed = tf.image.rgb_to_grayscale(self.frame)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(self.processed,
                                                [self.frame_height, self.frame_width],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    """
    @tf.function
    def __call__(self, frame):
        frame = tf.reshape(frame, [210, 160, 3])
        frame = tf.cast(frame, tf.uint8)
        processed = tf.image.rgb_to_grayscale(frame)
        processed = tf.image.crop_to_bounding_box(processed, 34, 0, 160, 160)
        processed = tf.image.resize(processed, [self.frame_height, self.frame_width],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return processed
    """
    def __call__(self, session, frame):
        """
        Args:
            session: A Tensorflow session object
            frame: A (210, 160, 3) frame of an Atari game in RGB
        Returns:
            A processed (84, 84, 1) frame in grayscale
        """
        return session.run(self.processed, feed_dict={self.frame:frame})


class DQN(object):
    """Implements a Deep Q Network"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
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

        self.input = tf.compat.v2.keras.Input(shape=(self.frame_height,
                                           self.frame_width, self.agent_history_length), dtype=tf.float32)
        #self.input = tf.placeholder(shape=[None, self.frame_height,
        #                                   self.frame_width, self.agent_history_length],
        #                            dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input/255

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4,
                                            kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')(self.inputscaled)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2,
                                            kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')(self.conv1)
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1,
                                            kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')(self.conv2)
        self.conv4 =tf.keras.layers.Conv2D(filters=hidden, kernel_size=[7, 7], strides=1,
                                           kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                           padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')(self.conv3)

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.keras.layers.Flatten()(self.valuestream)
        self.advantagestream = tf.keras.layers.Flatten()(self.advantagestream)
        self.advantage = tf.keras.layers.Dense(units=self.n_actions,
                                               kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                               name="advantage")(self.advantagestream)
        self.value = tf.keras.layers.Dense(units=1,
                                           kernel_initializer=tf.compat.v2.initializers.VarianceScaling(scale=2),
                                           name='value')(self.valuestream)

        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        #self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.target_q = tf.compat.v2.keras.Input(shape=(), dtype=tf.float32)
        # Action that was performed
        self.action = tf.compat.v2.keras.Input(shape=(), dtype=tf.int32)
        # self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)), axis=1)

        # Parameter updates
        self.loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)


class ExplorationExploitationScheduler(object):
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    """Компромисс разведки и эксплуатации"""

    def __init__(self, DQN, n_actions, eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=1000000,
                 replay_memory_start_size=50000, max_frames=25000000):
        """
        Args:
            DQN: A DQN object
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_frames

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        self.DQN = DQN

    def get_action(self, session, frame_number, state, evaluation=False):
        """
        Args:
            session: A tensorflow session object
            frame_number: Integer, number of the current frame
            state: A (84, 84, 4) sequence of frames of an Atari game in grayscale
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions - 1 determining the action the agent perfoms next
        """
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2*frame_number + self.intercept_2

        if np.random.rand(1) < eps:                                 # 𝜖 -  коэффициэнт компромисса разведки и эксплуатации (при 1 - только исследование)
            return np.random.randint(0, self.n_actions)             # Случайное число от 0 до максимального количества действий
        return session.run(self.DQN.best_action, feed_dict={self.DQN.input:[state]})[0]


class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    """Память для воспроизведения"""
    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

    def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
        """
        Args:
            session: A tensorflow sesson object
            replay_memory: A ReplayMemory object
            main_dqn: A DQN object
            target_dqn: A DQN object
            batch_size: Integer, Batch size
            gamma: Float, discount factor for the Bellman equation
        Returns:
            loss: The loss of the minibatch, for tensorboard
        Draws a minibatch from the replay memory, calculates the
        target Q-value that the prediction Q-value is regressed to.
        Then a parameter update is performed on the main DQN.
        """
        # Draw a minibatch from the replay memory
        states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
        # The main network estimates which action is best (in the next
        # state s', new_states is passed!)
        # for every transition in the minibatch
        arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
        # The target network estimates the Q-values (in the next state s', new_states is passed!)
        # for every transition in the minibatch
        q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
        double_q = q_vals[range(batch_size), arg_q_max]
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
        # if the game is over, targetQ=rewards
        target_q = rewards + (gamma*double_q * (1-terminal_flags))
        # Gradient descend step to update the parameters of the main network
        loss, _ = session.run([main_dqn.loss, main_dqn.update],
                              feed_dict={main_dqn.input:states,
                                         main_dqn.target_q:target_q,
                                         main_dqn.action:actions})
        return loss


class TargetNetworkUpdater(object):
    """Copies the parameters of the main DQN to the target DQN"""
    """Обновление параметров сети назначения"""
    def __init__(self, main_dqn_vars, target_dqn_vars):
        """
        Args:
            main_dqn_vars: A list of tensorflow variables belonging to the main DQN network
            target_dqn_vars: A list of tensorflow variables belonging to the target DQN network
        """
        self.main_dqn_vars = main_dqn_vars
        self.target_dqn_vars = target_dqn_vars

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.main_dqn_vars):
            copy_op = self.target_dqn_vars[i].assign(var.value())
            update_ops.append(copy_op)
        return update_ops

    def __call__(self, sess):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """
        update_ops = self._update_target_vars()
        for copy_op in update_ops:
            sess.run(copy_op)