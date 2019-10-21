import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import tensorflow as tf
import pandas as pd
from breakoutModel import MyModel
import pygame
import json
import requests


SCORE_REQUIMENT = 4
AGE = 1
FEATURES = ['frame', 'action']
INIT = True
TRAIN = False
SAVE = False
STEP = 1000
saved_model_path = "/home/konstantin/tf_models/breakout"

headers = {"content-type": "application/json"}


def initial(env, legal_actions, count_games, count_steps):
    accepted_scores = []
    training_data = []
    for each_game in range(count_games):
        score = 0
        game_memory = []
        prev_observation = []
        env.reset()
        for _ in range(count_steps):
            env.render()
            action = random.choice(legal_actions)
            observation, reward, done, info = env.step(action)
            if len(prev_observation):
                #game_memory.append([prev_observation, action])
                game_memory.append([(prev_observation.astype(np.float32)/255), action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score > SCORE_REQUIMENT * AGE:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0].tolist(), data[1]])
        env.reset()
    for_train = pd.DataFrame(training_data, columns=FEATURES)
    if not for_train.empty:
        print('Average accepted score:', mean(accepted_scores))
        # print('Median  - ', median(accepted_scores))
        # print(Counter(accepted_scores))
    env.close()
    return for_train


env = gym.make('Breakout-v0')
env.reset()
legal_actions = [0, 1, 2, 3]

model = MyModel()

if INIT:
    training_data = initial(env, legal_actions, 30, STEP)
    training_data.to_csv('saved.csv')

    if not training_data.empty:
        model.train(training_data)
    else:
        print('Not play witch ower Score requiment :')

test_data = initial(env, legal_actions, 10, STEP)
if not test_data.empty > 0:
    # test_data.to_csv('saved.csv')
    model.test(test_data)
if SAVE:
    path = model.save_model(saved_model_path)

"""
scores = []
choices = []
training_data = []

for each_game in range(1):
    score = 0
    game_memory = []
    prev_frame = []
    env.reset()
    for _ in range(STEP):
        if len(prev_frame) == 0:
            action = random.choice(legal_actions)
        else:
            # data = {'frame': [prev_frame.tolist()]}
            #action = model.predict(data)
            # print(prev_frame.astype(np.float32).tolist())
            data = json.dumps({"examples": [{"frame": prev_frame.astype(np.float32).tolist()}]})
            try:
                json_response = requests.post('http://localhost:8501/v1/models/breackout:classify',
                                              data=data, headers=headers)
            except:
                print("Not response from SavedModel in TensorFlow Serving")
                break
            predictions = json.loads(json_response.text)["results"][0]
            pred_list = dict(predictions)
            action = int(max(pred_list, key=pred_list.get))
            if _ % 10 == 0:
                action = 1


        # frame = ale.getRAM()
        if len(prev_frame):
            game_memory.append([prev_frame, action])
        choices.append(action)
        # score += ale.act(action)
        # prev_frame = frame
        #if ale.game_over():
            #break
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


