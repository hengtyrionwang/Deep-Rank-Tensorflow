'''
Descripttion:
version:
Author: Heng Tyrion Wang
Date: 2022-03-02 16:21:17
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 08:51:56
'''

import tensorflow as tf
from Layers.DeepNet import DeepNet
from Layers.FMLayer import FMLayer
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.LinearLayer import LinearLayer
from Layers.FeatureLayer import FeatureLayer

class DeepFM(tf.keras.Model):
    def __init__(self, options, name="DeepFM"):
        super(DeepFM, self).__init__(name=name)
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
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
        self.deep_prediction = tf.keras.layers.Dense(1)

        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)


    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        fm_output = self.fm_section([self.feature_embedding, feature_idx, feature_vals])
        fm_term = tf.math.reduce_sum(fm_output, 1, keepdims=True)

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        deep_input = tf.reshape(embeddings, [-1, self.field_dim*self.embedding_dim])
        deep_output = self.deep_section(deep_input, training=training)
        deep_term = self.deep_prediction(deep_output)

        combine_term = tf.math.add_n([linear_term, fm_term, deep_term])

        predictions = tf.math.sigmoid(combine_term)

        return predictions