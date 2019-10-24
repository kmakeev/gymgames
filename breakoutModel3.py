from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import datetime
import random
import gym
import numpy as np
from statistics import median, mean
import pandas as pd
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()
SCORE_REQUIMENT = 2
AGE = 1
STEP = 1000
MAX_LIVES = 5
FEATURES = ['count', 'lives', 'bricks', 'field', 'board', 'action']
NUM_CLASSES = 4
log_dir = "C:\\Python34\\gym\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def show_image(image):
    plt.imshow(image)
    plt.show()

def initial(env, legal_actions, count_games, max_len):
    accepted_scores = []
    training_data = []
    for each_game in range(count_games):
        score = 0
        prev_lives = MAX_LIVES                          #num lives as default
        score_by_life = 0
        prev_best_index = 0
        game_memory = []
        prev_observation = []
        env.reset()
        for count in range(STEP):
            # env.render()
            action = random.choice(legal_actions)
            observation, reward, done, info = env.step(action)
            # show_image(observation)
            gray_img = tf.compat.v2.image.rgb_to_grayscale(observation)
            # show_image(tf.squeeze(gray_img))
            squezze_gray_image = tf.squeeze(gray_img).numpy().astype(np.float32)    #/255
            top = squezze_gray_image[0:20]
            lives = info['ale.lives']
            bricks = squezze_gray_image[27:123]
            field = squezze_gray_image[123:188]
            board = squezze_gray_image[189:193]
            # show_image(squezze_gray_image[0:20])            # score and lives
            # show_image(squezze_gray_image[27:123])          # bricks
            # show_image(squezze_gray_image[123:188])         # field
            # show_image(board)  # board
            # print(score.shape, bricks.shape, field.shape, board.shape)
            if len(prev_observation):
                #game_memory.append([prev_observation, action])
                game_memory.append([float(count/STEP), float(lives/MAX_LIVES), bricks, field,
                                    board, action])
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
                training_data.append([data[0], data[1], data[2], data[3],
                                      data[4], data[5]])
                if len(training_data) == max_len:
                    break
        if len(training_data) == max_len:
            break
        env.reset()
    for_train = pd.DataFrame(training_data, columns=FEATURES)
    if not for_train.empty:
        print('Average accepted score:', mean(accepted_scores))
        # print('Median  - ', median(accepted_scores))
        # print(Counter(accepted_scores))
    env.close()
    return for_train


class MyModel():

    def __init__(self):

        input_count = tf.compat.v2.keras.Input(shape=(1, ), dtype='float32', name='count')
        input_lives = tf.compat.v2.keras.Input(shape=(1, ), dtype='float32', name='lives')

        input_bricks = tf.compat.v2.keras.Input(shape=(96, 160), dtype='float32', name='bricks')
        input_field = tf.compat.v2.keras.Input(shape=(65, 160), dtype='float32', name='field')
        input_board = tf.compat.v2.keras.Input(shape=(4, 160), dtype='float32', name='board')

        braunch1 = tf.keras.layers.Dense(1)(input_count)
        braunch2 = tf.keras.layers.Dense(1)(input_lives)

        #braunch3 = tf.keras.layers.Reshape((15360,))(input_bricks)
        braunch3 = tf.keras.layers.Conv1D(4, 10, activation='relu')(input_bricks)
        braunch3 = tf.keras.layers.Conv1D(2, 4, activation='relu')(braunch3)
        braunch3 = tf.keras.layers.Reshape((168, ))(braunch3)

        #braunch4 = tf.keras.layers.Reshape((10400,))(input_field)
        braunch4 = tf.keras.layers.Conv1D(2, 4,  activation='relu')(input_field)
        # braunch4 = tf.keras.layers.Conv1D(2, 2, activation='relu')(braunch4)
        braunch4 = tf.keras.layers.Reshape((124, ))(braunch4)

        braunch5 = tf.keras.layers.Conv1D(4, 4)(input_board)
        braunch5 = tf.keras.layers.Reshape((4, ))(braunch5)

        concotaneted = tf.keras.layers.concatenate([braunch1, braunch2, braunch3, braunch3, braunch4, braunch5])
        #concotaneted = tf.keras.layers.concatenate([braunch3, braunch3, braunch4, braunch5])
        concotaneted = tf.keras.layers.Reshape((233, 2))(concotaneted)
        last = tf.keras.layers.LSTM(16)(concotaneted)
        last = tf.keras.layers.LSTM(16)(concotaneted)
        last = tf.keras.layers.Dense(16, activation='relu')(last)
        last = tf.keras.layers.Dropout(0.2)(last)
        action = tf.keras.layers.Dense(4, activation='softmax', name='action')(last)
        self.model = tf.keras.Model([input_count, input_lives, input_bricks, input_field, input_board], action)
        #self.model = tf.keras.Model([input_bricks, input_field, input_board], action)

        self.model.compile(
            optimazer=tf.keras.optimizers.SGD(),
            loss='mse',
            metric=['acc']
        )
        tf.keras.utils.plot_model(self.model, show_shapes=True)
        self.model.summary()

    def input_from_dataframe(self, data):
        actions = data.pop('action').values
        inputs = {'count': data['count'].values, 'lives': data['lives'].values, 'bricks': np.concatenate(data['bricks'].values).reshape(-1, 96, 160),
                  'field': np.concatenate(data['field'].values).reshape(-1, 65, 160), 'board': np.concatenate(data['board'].values).reshape(-1, 4, 160)}
        return inputs, actions

    def train(self, training_data):
        train, train_y = self.input_from_dataframe(training_data)
        train_y = tf.keras.utils.to_categorical(train_y, NUM_CLASSES)
        tensoboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(x=train, y=train_y, epochs=5,
                       batch_size=100,
                       # validation_data=(test, test_y),
                       callbacks=[tensoboard_callback])

    def test(self, test_data):
        test, test_y = self.input_from_dataframe(test_data)
        test_y = tf.keras.utils.to_categorical(test_y, NUM_CLASSES)
        test_score = self.model.evaluate(test, test_y, verbose=2)
        print("Model evaluation result -", test_score)

    def predict(self, data):
        pred, pred_y = self.input_from_dataframe(data)
        result = self.model.predict(pred, batch_size=1)
        return result









