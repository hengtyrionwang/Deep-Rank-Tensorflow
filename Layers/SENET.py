'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 22:04:53
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 22:12:18
'''

import tensorflow as tf

class SENET(tf.keras.layers.Layer):
    def __init__(self, factor, field_dim, name="SENET"):
        super(SENET, self).__init__()
        self.factor = factor
        self.field_dim = field_dim
        self.reduction = int(self.field_dim/self.factor)

    def build(self, input_shape):
        self.W_1 = self.add_weight('w1', shape=[self.field_dim, self.reduction], initializer="random_normal", trainable=True)
        self.W_2 = self.add_weight('w2', shape=[self.reduction, self.field_dim], initializer="random_normal", trainable=True)

    def call(self, inputs, training=False):

        Z = tf.math.reduce_mean(inputs, axis=-1)

        A_1 = tf.nn.relu(tf.linalg.matmul(Z, self.W_1))
        A_2 = tf.nn.relu(tf.linalg.matmul(A_1, self.W_2))
        V = tf.math.multiply(inputs, tf.expand_dims(A_2, axis=2))
        return V