'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-12 15:41:50
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-12 15:41:50
'''

import tensorflow as tf
from Layers.CrossLayerV2 import CrossLayerV2

class CrossNetV2(tf.keras.layers.Layer):
    def __init__(self, units_num, dims, name="CrossNetV2"):
        super(CrossNetV2, self).__init__()
        self.units_num = units_num
        self.dims = dims
    
    def build(self, input_shape):
        self.cross_layers = []
        for i in range(self.units_num):
            self.cross_layers.append(CrossLayerV2(self.dims))

    def call(self, inputs, training=False):
        outputs = inputs
        for i in range(self.units_num):
            outputs = self.cross_layers[i](inputs, outputs)
        return outputs
