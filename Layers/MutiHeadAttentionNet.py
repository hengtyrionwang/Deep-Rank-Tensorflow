'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 20:27:19
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 20:36:57
'''
import tensorflow as tf
from Layers.MultiHeadAttentionLayer import MultiHeadAttentionLayer

class MultiHeadAttentionNet(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_units, num_heads, drop_rate, name="MultiHeadAttentionNet"):
        super(MultiHeadAttentionNet, self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        self.attention_layer = []
        for i in range(self.num_layers):
            self.attention_layer.append(MultiHeadAttentionLayer(self.num_units, self.num_heads, self.drop_rate))

    def call(self, inputs, training=False):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = self.attention_layer[i](outputs)
        return outputs