'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-12 15:57:57
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-12 16:17:13
'''

import tensorflow as tf

class CrossLayerMix(tf.keras.layers.Layer):
    def __init__(self, dims, name="CrossLayerMix"):
        super(CrossLayerMix, self).__init__(name=name)
        self.dims = dims

    def build(self, input_shape):
        self.project_dim = int(self.dims/2)
        self.u = tf.keras.layers.Dense(self.project_dim)
        self.v = tf.keras.layers.Dense(self.dims)

    def call(self, x0, x1):
        output = self.u(x1)
        output = self.v(output)
        output = tf.math.multiply(x0, output)
        output = tf.math.add(x1, output)
        return output