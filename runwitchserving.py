import gym
import random
import numpy as np
import json
import requests

"""
before running 
$> nohup tensorflow_model_server --rest_api_port=8501 --model_name=CartPole-v0 --model_base_path="/home/konstantin/tf_models/" >server.log 2>&1
"""


STEP = 5000

headers = {"content-type": "application/json"}


env = gym.make('CartPole-v0')
env.reset()

scores = []
choices = []

for each_game in range(10):
    score = 0
    prev_observation = []
    env.reset()
    for _ in range(STEP):
        env.render()

        if len(prev_observation) == 0:
            action = random.randrange(0, 2)
        else:
            data = json.dumps({"examples": [{"observation": prev_observation.tolist()}]})
            try:
                json_response = requests.post('http://localhost:8501/v1/models/CartPole-v0:classify',
                                  data=data, headers=headers)
            except:
                print("Not response from SavedModel in TensorFlow Serving")
                break
            predictions = json.loads(json_response.text)["results"][0]
            pred_list = dict(predictions)
            action = int(max(pred_list, key=pred_list.get))
            # action = random.randrange(0, 2)


        choices.append(action)
        observation, reward, done, info = env.step(action)
        prev_observation = observation
        score += reward
        if done:
            break
    scores.append(score)


print('Average Score:', sum(scores) / len(scores))
print('choice 1: {}  choice 0: {}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
env.close()