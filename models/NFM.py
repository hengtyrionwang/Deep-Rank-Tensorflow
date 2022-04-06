'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-04 16:06:31
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:06:45
'''

import tensorflow as tf
from Layers.DeepNet import DeepNet
from Layers.FMLayer import FMLayer
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.FeatureLayer import FeatureLayer

class NFM(tf.keras.Model):
    def __init__(self, options, name="NFM"):
        super(NFM, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.drop_rate = options.dropout_rate
        self.l2_reg = options.l2_reg
        self.hidden_units = eval(options.hidden_units)
  
    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.fm_section = FMLayer(self.feature_dim)
        self.deep_section = DeepNet(self.hidden_units,  self.drop_rate, None)
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.dense_section = tf.keras.layers.Dense(1)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)
        self.feature_bias = self.add_weight("feature_bias", shape=[self.feature_dim, 1], initializer="zeros", trainable=True)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        fm_output = self.fm_section([self.feature_embedding, feature_idx, feature_vals])

        deep_output = self.deep_section(fm_output, training=training)

        bilinear = self.dense_section(deep_output)

        feature_bias = tf.nn.embedding_lookup_sparse(self.feature_bias, feature_idx, None, combiner="sum")

        output = tf.math.add_n([bilinear, feature_bias])

        predictions = tf.math.sigmoid(output)

        return predictions
