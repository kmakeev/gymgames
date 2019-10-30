from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class FrameProcessor(object):
    """Resizes and converts RGB Atari frames to grayscale"""
    def __init__(self, frame_height=84, frame_width=84):
        """
        Args:
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
        """
        self.frame_height = frame_height
        self.frame_width = frame_width

    @tf.function
    def __call__(self, frame):
        frame = tf.reshape(frame, [210, 160, 3])
        frame = tf.cast(frame, tf.uint8)
        processed = tf.image.rgb_to_grayscale(frame)
        processed = tf.image.crop_to_bounding_box(processed, 34, 0, 160, 160)
        processed = tf.image.resize(processed, [self.frame_height, self.frame_width],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return processed
