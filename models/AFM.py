'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-05 07:56:56
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:06:43
'''

import tensorflow as tf
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.AttentionLayer import AttentionLayer
from Layers.FeatureLayer import FeatureLayer

class AFM(tf.keras.Model):
    def __init__(self, options, name="AFM"):
        super(AFM, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.drop_rate = options.dropout_rate
        self.l2_reg = options.l2_reg
        self.hidden_factors = [self.embedding_dim*2, self.embedding_dim]

    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.embedding_section = EmbeddingLayer(self.field_dim, None)
        self.dense_section = tf.keras.layers.Dense(1)

        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)
        self.feature_bias = self.add_weight("feature_bias", shape=[self.feature_dim, 1], initializer="zeros", trainable=True)
        self.attention = AttentionLayer(self.field_dim, self.embedding_dim, self.hidden_factors, self.drop_rate,self.regularizer)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        fm_input = tf.reshape(embeddings, [-1, self.field_dim, self.embedding_dim])

        afm = self.attention(fm_input, training=training)       

        bilinear = self.dense_section(afm)

        feature_bias = tf.nn.embedding_lookup_sparse(self.feature_bias, feature_idx, None, combiner="sum")

        output = tf.math.add_n([bilinear, feature_bias])

        predictions = tf.math.sigmoid(output)

        return predictions


