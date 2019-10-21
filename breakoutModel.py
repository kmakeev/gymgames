from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
os.environ['CUDA_VISIBLE_DEVICES'] = ''

FEATURES = ['frame', 'action']


class MyModel:

    def __init__(self):
        self.my_feature_columns = []
        self.my_feature_columns.append(tf.feature_column.numeric_column(key=FEATURES[0], shape=[210, 160, 3]))
        self.classifier = tf.compat.v2.estimator.DNNClassifier(
            feature_columns=self.my_feature_columns,
            hidden_units=[128, 4, ],
            n_classes=4,
            model_dir='./output')

    def save_model(self, patch):
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            tf.feature_column.make_parse_example_spec(self.my_feature_columns))
        export_path = self.classifier.export_saved_model(patch, serving_input_fn)
        return export_path


    def input_from_dataframe(self, data):
        # print(data.head())
        # print(data.dtypes)
        actions = data.pop('action')
        return data.to_dict('list'), actions.tolist()

    def input_fn(self, features, labels, training=True, batch_size=2048):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if training:
            dataset = dataset.shuffle(10).repeat()
        return dataset.batch(batch_size)

    def input_prediction_fn(self, features, batch_size=1):
        return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

    def train(self, training_data):
        train, train_y = self.input_from_dataframe(training_data)

        self.classifier.train(
            input_fn=lambda: self.input_fn(train, train_y),
            steps=100000)

    def test(self, test_data):
        test, test_y = self.input_from_dataframe(test_data)
        eval_result = self.classifier.evaluate(
            input_fn=lambda: self.input_fn(test, test_y, training=False))
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    def predict(self, data):

        predictions = self.classifier.predict(
            input_fn=lambda: self.input_prediction_fn(data), yield_single_examples=True)
        for pred_dict in predictions:
            action = pred_dict['class_ids'][0]
            # probability = pred_dict['probabilities'][action]
        return action