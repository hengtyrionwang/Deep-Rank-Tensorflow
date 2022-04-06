'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-12 15:12:50
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-12 16:17:17
'''

import tensorflow as tf

class CrossLayerV2(tf.keras.layers.Layer):
    def __init__(self, dims, name="CrossLayerV2"):
        super(CrossLayerV2, self).__init__(name=name)
        self.dims = dims

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.dims)

    def call(self, x0, x1):
        output = tf.math.multiply(x0, self.dense(x1))
        output = tf.math.add(x1, output)
        return output