import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import pandas as pd
from breakoutfirst.MyModel import MyModel


SCORE_REQUIMENT = 50
AGE = 4
FEATURES = ['observation', 'action']
INIT = False
TRAIN = False
SAVE = False
STEP = 5000
saved_model_path = "c:\\Python34\\gym\\CartPoleModel"


def initial(env, count_games, count_steps):
    training_data = []
    accepted_scores = []
    for _ in range(count_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(count_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation):
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        if score >= SCORE_REQUIMENT:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0].tolist(), data[1]])
        env.reset()
    for_train = pd.DataFrame(np.array(training_data), columns=FEATURES)
    print('Average accepted score:', mean(accepted_scores))
    print('Median  - ', median(accepted_scores))
    print(Counter(accepted_scores))

    return for_train


env = gym.make('CartPole-v0')
env.reset()
# training_data.to_csv('saved.csv')
model = MyModel()

if INIT:

    training_data = initial(env, 10000, 5000)
    model.train(training_data)

test_data = initial(env, 100, 50)
model.test(test_data)

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
            action = random.randrange(0, 2)
        else:
            data = {'observation': [prev_observation.tolist()]}
            action = model.predict(data)
            game_memory.append([observation, action])

        choices.append(action)
        observation, reward, done, info = env.step(action)
        prev_observation = observation
        score += reward
        if done:
            break
    if score >= SCORE_REQUIMENT * AGE:
        for data in game_memory:
            training_data.append([data[0].tolist(), data[1]])
    scores.append(score)


print('Average Score:', sum(scores) / len(scores))
print('choice 1: {}  choice 0: {}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(SCORE_REQUIMENT)
env.close()

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


