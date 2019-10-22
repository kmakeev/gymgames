from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

FEATURES = ['count', 'lives', 'bricks', 'field', 'board', 'action']

input_count = tf.compat.v2.keras.Input(shape=(1, ), dtype='float32', name='count')
input_lives = tf.compat.v2.keras.Input(shape=(1, ), dtype='float32', name='lives')

input_bricks = tf.compat.v2.keras.Input(shape=(96, 160), dtype='float32', name='bricks')
input_field = tf.compat.v2.keras.Input(shape=(65, 160), dtype='float32', name='field')
input_board = tf.compat.v2.keras.Input(shape=(4, 160), dtype='float32', name='board')

braunch1 = tf.keras.layers.Dense(1)(input_count)
braunch2 = tf.keras.layers.Dense(1)(input_lives)

braunch3 = tf.keras.layers.Reshape((15360,))(input_bricks)
braunch3 = tf.keras.layers.Dense(128)(braunch3)
braunch3 = tf.keras.layers.Dense(32, activation='relu')(braunch3)

braunch4 = tf.keras.layers.Reshape((10400,))(input_field)
braunch4 = tf.keras.layers.Dense(64)(braunch4)
braunch4 = tf.keras.layers.Dense(16, activation='relu')(braunch4)

braunch5 = tf.keras.layers.Reshape((640,))(input_board)
braunch5 = tf.keras.layers.Dense(8)(braunch5)

concotaneted = tf.keras.layers.concatenate([braunch1, braunch2, braunch3, braunch4, braunch5])
concotaneted = tf.keras.layers.Reshape((29, 2))(concotaneted)
concotaneted = tf.keras.layers.LSTM(8)(concotaneted)
action = tf.keras.layers.Dense(4, activation='softmax', name='action')(concotaneted)

model = tf.keras.Model([input_count, input_lives, input_bricks, input_field, input_board], action)
tf.keras.utils.plot_model(model, show_shapes=True)
model.summary()





