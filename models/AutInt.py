'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-04 16:07:03
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:06:41
'''

import tensorflow as tf
from Layers.DeepNet import DeepNet
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.LinearLayer import LinearLayer
from Layers.MutiHeadAttentionNet import MultiHeadAttentionNet
from Layers.FeatureLayer import FeatureLayer

class AutoInt(tf.keras.Model):
    def __init__(self, options, name="AutoInt"):
        super(AutoInt, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.num_units = options.num_units
        self.num_heads = options.num_heads
        self.num_atten = options.num_atten
        self.drop_rate = options.dropout_rate
        self.l2_reg = options.l2_reg
        self.hidden_units = eval(options.hidden_units)

    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.deep_section = DeepNet(self.hidden_units,  self.drop_rate, None)
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
        self.deep_prediction = tf.keras.layers.Dense(1)
        self.attention_projection = tf.keras.layers.Dense(1)
        self.attention_section = MultiHeadAttentionNet(self.num_atten, self.num_units, self.num_heads, self.drop_rate)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        deep_input = tf.reshape(embeddings, [-1, self.field_dim*self.embedding_dim])
        deep_output = self.deep_section(deep_input, training=training)
        deep_term = self.deep_prediction(deep_output)

        attention_input = tf.reshape(embeddings, [-1, self.field_dim, self.embedding_dim])
        attention_out = self.attention_section(attention_input, training=training)
        attention_out = tf.reshape(attention_out, [-1, self.field_dim*self.num_units])
        attention_term = self.attention_projection(attention_out)

        combine_term = tf.math.add_n([linear_term, deep_term, attention_term])

        predictions = tf.math.sigmoid(combine_term)

        return predictions