'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-21 08:25:31
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:45:52
'''

import tensorflow as tf

class ExtremeFMLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, field_dim0, field_dim2, layer_size, name="ExtremeFMLayer"):
        super(ExtremeFMLayer, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.field_dim1 = field_dim0
        self.field_dim2 = field_dim2
        self.layer_size = layer_size

    def build(self, input_shape):
        self.filters = self.add_weight('filters', shape=[1, self.field_dim1*self.field_dim2, self.layer_size],
                                initializer="random_normal", trainable=True)

    def call(self, x0, x1):
        split_tensor0 = tf.split(x0, self.embedding_dim * [1], 2)
        split_tensor1 = tf.split(x1, self.embedding_dim * [1], 2)
        dot_result_m = tf.linalg.matmul(split_tensor0, split_tensor1, transpose_b=True)
        dot_result_o = tf.reshape(dot_result_m, shape=[self.embedding_dim, -1, self.field_dim1*self.field_dim2])
        dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
        curr_out = tf.nn.conv1d(dot_result, filters=self.filters, stride=1, padding='VALID')
        curr_out = tf.transpose(curr_out, perm=[0, 2, 1])
        return curr_out