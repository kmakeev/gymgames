from DQN import DQN, ReplayMemory, ExplorationExploitationScheduler, FrameProcessor, TargetNetworkUpdater
from atari import Atari
import imageio
from skimage.transform import resize
import tensorflow.compat.v1 as tf
import numpy as np
import os


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
    """Двойное Q-Learning"""
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


def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)

    imageio.mimsave(f'{path}{"ATARI_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)


def clip_reward(reward):
    """Отсечение наград.
    Поскольку шкала оценок сильно варьируется от игры к игре, принято все положительные награды равны 1,
    а все отрицательные награды равны -1,
    нулевые награды без изменений"""
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1


tf.reset_default_graph()

# Global parameters train or learn
TRAIN = True
# Environments game name
ENV_NAME = 'BreakoutDeterministic-v4'

# Control parameters
# Максимальное количество кадров для одной игры
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
# Количество кадров считываемое агентов между оценками
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
# Количество кадров для одной оценки
EVAL_STEPS = 10000               # Number of frames for one evaluation
# Количество выбранных действий между обновлениями целевой сети
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
# According to Mnih et al. 2015 this is measured in the number of
# parameter updates (every four actions), however, in the
# DeepMind code, it is clearly measured in the number
# of actions the agent choses
# 𝛾 - коэффициент дисконтирования
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation

# Количество совершенно случайных действий, прежде чем агент начнет обучение
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
# before the agent starts learning
# Максимальное количество фреймой которые агент видит
MAX_FRAMES = 30000000            # Total number of frames the agent sees
# Количество переходов, хранящихся в памяти воспроизведения
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
# Количество действий «NOOP» или «FIRE» в начале
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
# evaluation episode
# Каждые четыре действия выполняется шаг градиентного спуска
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
# has the shape (1,1,1024) which is split into two streams. Both
# the advantage stream and value stream have the shape
# (1,1,512). This is slightly different from the original
# implementation but tests I did with the environment Pong
# have shown that this way the score increases more quickly
# Количество фильтров в конечном сверточном слое. Выход
# имеет форму (1,1,1024), которай разбита на два потока:
# поток преимуществ и поток создания ценности, которые  имеют форму
# (1,1,512).
# Cкорость обучения для оптимизатора Adam
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
# Hessel et al. 2017 used 0.0000625
# Размер пачки для обучения
BS = 32                          # Batch size
# For compatibility
tf.disable_eager_execution()
PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))

# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)   # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               # (★★)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

LAYER_IDS = ["conv1", "conv2", "conv3", "conv4", "denseAdvantage",
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalar summaries for tensorboard: loss, average reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramm summaries for tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)


def train():
    """Contains the training and evaluation loops"""
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)   # (★)
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        sess.run(init)

        frame_number = 0
        rewards = []
        loss_list = []

        while frame_number < MAX_FRAMES:

            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                terminal_life_lost = atari.reset(sess)
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    # (4★)
                    atari.env.render()
                    action = explore_exploit_sched.get_action(sess, frame_number, atari.state)
                    # (5★)
                    processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(sess, action)
                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    # Clip the reward
                    clipped_reward = clip_reward(reward)

                    # (7★) Store transition in the replay memory сохранить переход в памяти replay памяти
                    my_replay_memory.add_experience(action=action,
                                                    frame=processed_new_frame[:, :, 0],
                                                    reward=clipped_reward,
                                                    terminal=terminal_life_lost)
                    # При большем количестве случайных действий ( REPLAY_MEMORY_START_SIZE)
                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE: # Каждые четыре действия и  выполняется шаг градиентного спуска
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)
                        loss_list.append(loss)
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE: # Количество выбранных действий между обновлениями целевой сети.
                        update_networks(sess) # (9★)

                    if terminal:            # при окончании игры
                        terminal = False
                        break

                rewards.append(episode_reward_sum)              # Набрано очков за игру

                # Output the progress:
                if len(rewards) % 10 == 0:                      #каждую 10-ю игру
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:         #Каждые 10 проигрышей при большем количестве случайных действий ( REPLAY_MEMORY_START_SIZE= 10000)
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH:np.mean(loss_list),
                                                   REWARD_PH:np.mean(rewards[-100:])})

                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)                  #Для сводки на Tensorboard
                    SUMM_WRITER.add_summary(summ_param, frame_number)

                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number,
                              np.mean(rewards[-100:]), file=reward_file)

            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(EVAL_STEPS):
                if terminal:
                    terminal_life_lost = atari.reset(sess, evaluation=True)
                    episode_reward_sum = 0
                    terminal = False

                # Fire (action 1), when a life was lost or the game just started,
                # so that the agent does not stand around doing nothing. When playing
                # with other environments, you might want to change this...
                action = 1 if terminal_life_lost else explore_exploit_sched.get_action(sess, frame_number,
                                                                                       atari.state,
                                                                                       evaluation=True)

                processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif:
                    frames_for_gif.append(new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif

            print("Evaluation score:\n", np.mean(eval_rewards))
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")

            #Save the network parameters
            saver.save(sess, PATH+'/my_model', global_step=frame_number)
            frames_for_gif = []

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)

if TRAIN:
    train()

save_files_dict = {
    'BreakoutDeterministic-v4':("trained/breakout/", "my_model-7016868.meta"),
    'PongDeterministic-v4':("trained/pong/", "my_model-3217770.meta")
}

if not TRAIN:

    gif_path = "GIF/"
    os.makedirs(gif_path, exist_ok=True)

    trained_path, save_file = save_files_dict[ENV_NAME]

    explore_exploit_sched = ExplorationExploitationScheduler(
        MAIN_DQN, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(trained_path+save_file)
        saver.restore(sess,tf.train.latest_checkpoint(trained_path))
        frames_for_gif = []
        terminal_life_lost = atari.reset(sess, evaluation = True)
        episode_reward_sum = 0
        while True:
            atari.env.render()
            action = 1 if terminal_life_lost else explore_exploit_sched.get_action(sess, 0, atari.state,
                                                                                   evaluation = True)

            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(sess, action)
            episode_reward_sum += reward
            frames_for_gif.append(new_frame)
            if terminal == True:
                break

        atari.env.close()
        print("The total reward is {}".format(episode_reward_sum))
        print("Creating gif...")
        generate_gif(0, frames_for_gif, episode_reward_sum, gif_path)
        print("Gif created, check the folder {}".format(gif_path))