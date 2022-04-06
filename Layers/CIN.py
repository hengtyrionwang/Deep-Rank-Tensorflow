'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-21 07:50:27
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:45:34
'''

import tensorflow as tf
from Layers.ExtremeFMLayer import ExtremeFMLayer

class CIN(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, field_dim, layer_size, name="CIN"):
        super(CIN, self).__init__()
        self.embedding_dim = embedding_dim
        self.field_dim = field_dim
        self.layer_size = layer_size
    
    def build(self, input_shape):
        self.exFM = []
        next_field_dim = self.field_dim
        for i in range(len(self.layer_size)):
            self.exFM.append(ExtremeFMLayer(self.embedding_dim, self.field_dim, next_field_dim, self.layer_size[i]))
            next_field_dim = int(self.layer_size[i])
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        outputs = inputs
        results = []
        for i in range(len(self.layer_size)):
            outputs = self.exFM[i](inputs, outputs)
            results.append(outputs)
        final_result = tf.concat(results, axis=1)
        final_result = tf.reduce_sum(final_result, -1)
        exFM = self.dense(final_result)
        return exFM