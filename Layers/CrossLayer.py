'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 17:16:33
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 22:27:14
'''

import tensorflow as tf

class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, dims, name="CrossLayer"):
        super(CrossLayer, self).__init__(name=name)
        self.dims = dims

    def build(self, input_shape):

        self.weight = self.add_weight('w', shape=[self.dims,1],
                                      initializer="random_normal", trainable=True)
        self.bias = self.add_weight('b', [self.dims], initializer="zeros", trainable=True)

    def call(self, x0, x1):
        x_1w = tf.linalg.matmul(x1, self.weight)
        cross_terms = tf.math.multiply(x0, x_1w)
        output = tf.math.add(tf.math.add(cross_terms, self.bias), x1)
        return output