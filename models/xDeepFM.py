'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-21 08:59:56
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 08:54:00
'''

import tensorflow as tf
from Layers.CIN import CIN
from Layers.DeepNet import DeepNet
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.LinearLayer import LinearLayer
from Layers.FeatureLayer import FeatureLayer

class xDeepFM(tf.keras.Model):
    def __init__(self, options, name="xDeepFM"):
        super(xDeepFM, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.drop_rate = options.dropout_rate
        self.l2_reg = options.l2_reg
        self.num_cross = options.num_cross
        self.hidden_units = eval(options.hidden_units)
        self.layer_size = eval(options.layer_size)

    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)
        self.deep_section = DeepNet(self.hidden_units,  self.drop_rate, None)
        self.CIN_section = CIN(self.embedding_dim, self.field_dim, self.layer_size)
        self.deep_prediction = tf.keras.layers.Dense(1)
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        deep_input = tf.reshape(embeddings, [-1, self.field_dim*self.embedding_dim])
        cin_input = tf.reshape(embeddings, [-1, self.field_dim, self.embedding_dim])

        deep_output = self.deep_section(deep_input, training=training)
        deep_term = self.deep_prediction(deep_output)
        
        cin_term = self.CIN_section(cin_input)
        combine_term = tf.math.add_n([linear_term, cin_term, deep_term])


        predictions = tf.math.sigmoid(combine_term)

        return predictions