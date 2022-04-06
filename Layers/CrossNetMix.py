'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-12 16:03:22
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-12 16:06:53
'''

import tensorflow as tf
from Layers.CrossLayerMix import CrossLayerMix

class CrossNetMix(tf.keras.layers.Layer):
    def __init__(self, units_num, dims, name="CrossNetMix"):
        super(CrossNetMix, self).__init__()
        self.units_num = units_num
        self.dims = dims
    
    def build(self, input_shape):
        self.cross_layers = []
        for i in range(self.units_num):
            self.cross_layers.append(CrossLayerMix(self.dims))

    def call(self, inputs, training=False):
        outputs = inputs
        for i in range(self.units_num):
            outputs = self.cross_layers[i](inputs, outputs)
        return outputs
