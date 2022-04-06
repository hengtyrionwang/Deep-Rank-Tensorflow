'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 17:16:39
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 17:22:53
'''

import tensorflow as tf
from Layers.CrossLayer import CrossLayer

class CrossNet(tf.keras.layers.Layer):
    def __init__(self, units_num, dims, name="CrossNet"):
        super(CrossNet, self).__init__()
        self.units_num = units_num
        self.dims = dims
    
    def build(self, input_shape):
        self.cross_layers = []
        for i in range(self.units_num):
            self.cross_layers.append(CrossLayer(self.dims))

    def call(self, inputs, training=False):
        outputs = inputs
        for i in range(self.units_num):
            outputs = self.cross_layers[i](inputs, outputs)
        return outputs
