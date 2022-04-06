'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-03 12:43:32
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:06:48
'''

import tensorflow as tf
from Layers.LinearLayer import LinearLayer
from Layers.FMLayer import FMLayer
from Layers.FeatureLayer import FeatureLayer

class FM(tf.keras.Model):
    def __init__(self, options, name="FM"):
        super(FM, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.l2_reg = options.l2_reg

    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.fm_section = FMLayer(self.feature_dim)
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        fm_output = self.fm_section([self.feature_embedding, feature_idx, feature_vals])
        fm_term = tf.math.reduce_sum(fm_output, 1, keepdims=True)

        combine_term = tf.math.add_n([linear_term, fm_term])

        predictions = tf.math.sigmoid(combine_term)

        return predictions

