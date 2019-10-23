import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import tensorflow as tf
import pandas as pd
from breakoutModel3 import MyModel, initial
import pygame
import json
import requests
import matplotlib.pyplot as plt

INIT = True
TRAIN = False
SAVE = True
SCORE_REQUIMENT = 3
AGE = 1
STEP = 1000
FEATURES = ['count', 'lives', 'bricks', 'field', 'board', 'action']
legal_actions = [0, 1, 2, 3]
saved_model_path = './kerasmodel/'

tf.compat.v1.enable_eager_execution()

env = gym.make('Breakout-v0')

if INIT:
    training_data = initial(env, legal_actions, 100, 20)
    test_data = initial(env, legal_actions, 50, 10)

    if len(training_data)==0 or len(test_data)==0:
        print("Not play witch ower Score requiment ")
        exit()

    model = MyModel()
    model.train(training_data)
    model.test(test_data)
if SAVE:
    model.model.save(saved_model_path, save_format='tf')


scores = []
choices = []

accepted_scores = []
training_data = []
max_len = 10000
for each_game in range(1):
    score = 0
    prev_lives = 5                          #num lives as default
    score_by_life = 0
    prev_best_index = 0
    game_memory = []
    prev_observation = []
    env.reset()
    for count in range(STEP):
        env.render()
        if len(prev_observation) == 0:
            action = random.choice(legal_actions)
        else:
            data = prev_data
            for_predict_data = [float(data[0]/count), float(data[1]/5), data[2].ravel(), data[3],
                                  data[4], data[5]]
            dataframe = pd.DataFrame([for_predict_data], columns=FEATURES)
            predictions = model.predict(dataframe)
            action = predictions[0].argmax()
            if count % 10 == 0:
                action = 1
        print("For count %s action is - %s" % (count, action))
        observation, reward, done, info = env.step(action)
        gray_img = tf.compat.v2.image.rgb_to_grayscale(observation)
        squezze_gray_image = tf.squeeze(gray_img).numpy().astype(np.float32)/255
        top = squezze_gray_image[0:20]
        lives = info['ale.lives']
        bricks = squezze_gray_image[27:123]
        field = squezze_gray_image[123:188]
        board = squezze_gray_image[188:192]

        prev_data = [count, lives, bricks, field, board, action]
        game_memory.append(prev_data)
        prev_observation = observation
        score += reward
        score_by_life += reward
        if prev_lives != lives:
            if score_by_life == 0:
                del game_memory[prev_best_index:]
            else:
                prev_best_index = len(game_memory)
            score_by_life = 0
        prev_lives = lives
        if done:
            break
    if score >= SCORE_REQUIMENT * AGE and len(training_data) < max_len:
        accepted_scores.append(score)
        for data in game_memory:                                        # 5- number of lives as default
            training_data.append([float(data[0]/count), float(data[1]/5), data[2].ravel(), data[3],
                                  data[4], data[5]])
            if len(training_data) == max_len:
                break
    if len(training_data) == max_len:
        break
    env.reset()

env.close()
