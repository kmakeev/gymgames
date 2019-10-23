import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import tensorflow as tf
import pandas as pd
from breakoutModel2 import MyModel
import pygame
import json
import requests
import matplotlib.pyplot as plt

SCORE_REQUIMENT = 4
AGE = 1
# FEATURES = ['frame', 'action']
FEATURES = ['count', 'lives', 'bricks', 'field', 'board', 'action']
INIT = False
TRAIN = False
SAVE = False
STEP = 1000
saved_model_path = "/home/konstantin/tf_models/breakout2"
# tf.enable_eager_execution()


headers = {"content-type": "application/json"}


def show_image(image):
    plt.imshow(image)
    plt.show()

def initial(env, legal_actions, count_games, count_steps):
    accepted_scores = []
    training_data = []
    for each_game in range(count_games):
        score = 0
        game_memory = []
        prev_observation = []
        env.reset()
        for count in range(count_steps):
            # env.render()
            action = random.choice(legal_actions)
            observation, reward, done, info = env.step(action)
            # show_image(observation)
            gray_img = tf.compat.v2.image.rgb_to_grayscale(observation)
            # show_image(tf.squeeze(gray_img))
            squezze_gray_image = tf.squeeze(gray_img).numpy().astype(np.float32)/255
            top = squezze_gray_image[0:20]
            lives = info['ale.lives']
            bricks = squezze_gray_image[27:123]
            field = squezze_gray_image[123:188]
            board = squezze_gray_image[188:192]
            # show_image(squezze_gray_image[0:20])            # score and lives
            # show_image(squezze_gray_image[27:123])          # bricks
            # show_image(squezze_gray_image[123:188])         # field
            # show_image(squezze_gray_image[188:192])  # board
            # print(score.shape, bricks.shape, field.shape, board.shape)
            if len(prev_observation):
                #game_memory.append([prev_observation, action])
                game_memory.append([count, lives, bricks, field,
                                    board, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= SCORE_REQUIMENT * AGE:
            accepted_scores.append(score)
            for data in game_memory:                                        # 5- number of lives as default
                training_data.append([float(data[0]/count), float(data[1]/5), data[2].tolist(), data[3].tolist(),
                                      data[4].tolist(), data[5]])
        env.reset()
    for_train = pd.DataFrame(training_data, columns=FEATURES)
    if not for_train.empty:
        print('Average accepted score:', mean(accepted_scores))
        # print('Median  - ', median(accepted_scores))
        # print(Counter(accepted_scores))
    env.close()
    return for_train


env = gym.make('Breakout-v0')
legal_actions = [0, 1, 2, 3]

model = MyModel()
if INIT:
    training_data = initial(env, legal_actions, 50, STEP)
    # training_data.to_csv('saved.csv')

    if not training_data.empty:
        model.train(training_data)
    else:
        print('Not play witch ower Score requiment :')

test_data = initial(env, legal_actions, 10, STEP)
if not test_data.empty > 0:
    pass
    # test_data.to_csv('saved.csv')
    #model.test(test_data)
if SAVE:
    path = model.save_model(saved_model_path)


scores = []
choices = []
training_data = []
for each_game in range(1):
    score = 0
    game_memory = []
    prev_observation = []
    env.reset()
    for _ in range(STEP):
        env.render()
        if len(prev_observation) == 0:
            action = random.choice(legal_actions)
        else:
            ttt = prev_observation.reshape(100800,).astype(np.float32)
            data = json.dumps({"examples": [{"frame": ttt.tolist()}]})
            try:
                json_response = requests.post('http://localhost:8501/v1/models/breackout:classify',
                                              data=data, headers=headers)
            except:
                print("Not response from SavedModel in TensorFlow Serving")
                break
            if json_response.status_code != 200:
                print("Response code is not 200", json_response.text)
                break
            predictions = json.loads(json_response.text)["results"][0]
            pred_list = dict(predictions)
            action = int(max(pred_list, key=pred_list.get))
            if _ % 10 == 0:
                action = 2
        observation, reward, done, info = env.step(action)
        score += reward
        prev_observation = observation
        game_memory.append([prev_observation, action])
        choices.append(action)
        if done:
            break

    if score >= SCORE_REQUIMENT * AGE:
        for data in game_memory:
            training_data.append([data[0].tolist(), data[1]])
    scores.append(score)
    #ale.reset_game()
print('Average Score:', sum(scores) / len(scores))
print('choice 0: {}  choice 1: {}   choice 2: {}   choice 3: {} choice 4: {}'.format(choices.count(0) / len(choices), choices.count(1) / len(choices),
                                                                                     choices.count(2) / len(choices), choices.count(3) / len(choices),
                                                                                     choices.count(4) / len(choices)))
print(SCORE_REQUIMENT)
if len(training_data) and TRAIN:
    print("Traning...")
    for_train = pd.DataFrame(np.array(training_data), columns=FEATURES)
    model.train(for_train)
    print("DONE!")

"""
# model = model.classifier.get_model()  # get a fresh model
# saved_model_path = "/tmp/tf_save"
# tf.saved_model.save(model, saved_model_path)
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = random.randrange(0, 2)
        observation, reward, done, info = env.step(action)
        if done:
            print("Finish after {} timesteps".format(t+1))
            break
env.close()
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=CartPole-v0 \
  --model_base_path="/home/konstantin/tf_models/" >server.log 2>&1
"""
