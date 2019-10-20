import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import tensorflow as tf
import pandas as pd
from breakoutModel import MyModel
import pygame
from ale_python_interface import ALEInterface
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

labels = [
    "noop",
    "fire",
    "up",
    "right",
    "left",
    "down",
    "upright",
    "upleft",
    "downright",
    "downleft",
    "upfire",
    "rightfire",
    "leftfire",
    "downfire",
    "uprightfire",
    "upleftfire",
    "downrightfire",
    "downleftfire",
]

def initial(ale, count_games, count_steps):
    training_data = []
    accepted_scores = []
    legal_actions = ale.getMinimalActionSet()
    for _ in range(count_games):
        score = 0
        game_memory = []
        prev_frame = []
        for _ in range(count_steps):
            action = random.choice(legal_actions)
            # (screen_width, screen_height) = ale.getScreenDims()
            # frame = np.zeros(screen_width * screen_height, dtype=np.uint8)
            frame = ale.getRAM()
            # frame = np.array(frame, dtype=float)
            score += ale.act(action)
            if len(prev_frame):
                game_memory.append([prev_frame, action])
            prev_frame = frame
            if ale.game_over():
                break
        if score >= SCORE_REQUIMENT:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0].astype(np.float32).tolist(), data[1]])
        ale.reset_game()
    for_train = pd.DataFrame(training_data, columns=FEATURES)
    if not for_train.empty:
        print('Average accepted score:', mean(accepted_scores))
    # print('Median  - ', median(accepted_scores))
    # print(Counter(accepted_scores))

    return for_train

pygame.init()

ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.setBool(b'display_screen', False)
ale.loadROM(b'breakout.a26')

model = MyModel()

if INIT:
    training_data = initial(ale, 100, STEP)
    #training_data.to_csv('saved.csv')

    if not training_data.empty:
        model.train(training_data)
    else:
        print('Not play witch ower Score requiment :')

test_data = initial(ale, 10, STEP)
if not test_data.empty > 0:
    #test_data.to_csv('saved.csv')
    model.test(test_data)
if SAVE:
    path = model.save_model(saved_model_path)

scores = []
choices = []
training_data = []
legal_actions = ale.getMinimalActionSet()
ale.setBool(b'display_screen', True)
ale.loadROM(b'breakout.a26')

for each_game in range(1):
    score = 0
    game_memory = []
    prev_frame = []
    ale.reset_game()
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


        frame = ale.getRAM()
        if len(prev_frame):
            game_memory.append([prev_frame, action])
        choices.append(action)
        score += ale.act(action)
        prev_frame = frame
        if ale.game_over():
            break
    if score >= SCORE_REQUIMENT * AGE:
        for data in game_memory:
            training_data.append([data[0].tolist(), data[1]])
    scores.append(score)
    ale.reset_game()

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

"""

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


