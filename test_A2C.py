from Algorithms.A2C import A2C
from atari import Atari
from gym.spaces import Box, Discrete, Tuple
import os
import imageio
from skimage.transform import resize
import numpy as np
import tensorflow as tf


from ReplayMemory import ReplayMemory
from ExplorationExploitationScheduler import ExplorationExploitationScheduler

# tf.compat.v1.enable_eager_execution()
# Global parameters train or learn
TRAIN = True
# Environments game name
# ENV_NAME = 'BreakoutDeterministic-v4'
# ENV_NAME = 'PongDeterministic-v4'
ENV_NAME = 'MsPacmanDeterministic-v4'
# Control parameters
# Максимальное количество кадров для одной игры
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
# Количество кадров считываемое агентов между оценками
EVAL_FREQUENCY = 200000          # Number of frames the agent sees between evaluations
# Количество кадров для записи GIF одной эволюции
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
REPLAY_MEMORY_START_SIZE = 50000  # Number of completely random actions,
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
TAU = 0.08                       # The merging rate of the weight values between the primary and target networks
# Hessel et al. 2017 used 0.0000625
# Размер пачки для обучения
BS = 32                          # Batch size
# For compatibility
PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
# os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
# SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))

atari = Atari(ENV_NAME, NO_OP_STEPS)

print("The environment has the following {} actions: {}".format(atari.env.action_space.n,
                                                                atari.env.unwrapped.get_action_meanings()))
MAX_EPISODE = 10000
BASE_DIR = os.path.join('c:\\Python34\\Gym\\', 'A2CModel')

atari = Atari(ENV_NAME, NO_OP_STEPS)

if isinstance(atari.env.observation_space, Box):
    s_dim = atari.env.observation_space.shape[0] if len(atari.env.observation_space.shape) == 1 else 0
else:
    s_dim = int(atari.env.observation_space.n)

if len(atari.env.observation_space.shape) == 3:
    visual_sources = 1
    visual_resolution = list(atari.env.observation_space.shape)
else:
    visual_sources = 0
    visual_resolution = []

if isinstance(atari.env.action_space, Box):
    assert len(atari.env.action_space.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
    a_dim_or_list = atari.env.action_space.shape
    action_type = 'continuous'
elif isinstance(atari.env.action_space, Tuple):
    assert all([isinstance(i, Discrete) for i in
                atari.env.action_space]) == True, 'if action space is Tuple, each item in it must have type Discrete'
    a_dim_or_list = [i.n for i in atari.env.action_space]
    action_type = 'discrete'
else:
    a_dim_or_list = [atari.env.action_space.n]
    action_type = 'discrete'


model = A2C(
        s_dim=s_dim,
        visual_sources=visual_sources,
        visual_resolution=visual_resolution,
        a_dim_or_list=a_dim_or_list,
        action_type=action_type,
        max_episode=MAX_EPISODE,
        base_dir=BASE_DIR,
        logger2file=None,
        out_graph=None,
)


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


def learn(replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Args:
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
    model.learn(episode=episode, step=step)


def train():
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)  # (★)
    explore_exploit_sched = ExplorationExploitationScheduler(
        model, atari.env.action_space.n,
        replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)
    frame_number = 0
    rewards = []

    while frame_number < MAX_FRAMES:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0
        while epoch_frame < EVAL_FREQUENCY:
            loss_list = []
            terminal_life_lost = atari.reset()
            episode_reward_sum = 0
            for _ in range(MAX_EPISODE_LENGTH):
                # (4★)
                atari.env.render()
                action = explore_exploit_sched.get_action(frame_number, atari.state)
                # (5★)
                processed_new_frame, reward, terminal, terminal_life_lost, _ = atari.step(action)
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
                if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:  # Каждые четыре действия и  выполняется шаг градиентного спуска
                    loss = learn(my_replay_memory, BS)  # (8★)
                    loss_list.append(loss)

                if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:  # Количество выбранных действий между обновлениями целевой сети.
                    update_networks(MAIN_DQN, TARGET_DQN)  # (9★)


                if terminal:  # при окончании игры
                    terminal = False
                    break

            rewards.append(episode_reward_sum)  # Набрано очков за игру

            # Output the progress:
            if len(rewards) % 10 == 0:  # каждую 10-ю игру
                print(len(rewards), frame_number, np.mean(rewards[-100:]), np.mean(loss_list))
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
                terminal_life_lost = atari.reset(evaluation=True)
                episode_reward_sum = 0
                terminal = False
            atari.env.render()
            # Fire (action 1), when a life was lost or the game just started,
            # so that the agent does not stand around doing nothing. When playing
            # with other environments, you might want to change this...

            if terminal_life_lost:
                action = 1
            else:
                action = explore_exploit_sched.get_action(frame_number, atari.state, evaluation=True)

            processed_new_frame, reward, terminal, terminal_life_lost, new_frame = atari.step(action)
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
        atari.env.close()
        # Save the network parameters
        tf.saved_model.save(model, PATH+'my_model')
        # frames_for_gif = []

if TRAIN:
    train()