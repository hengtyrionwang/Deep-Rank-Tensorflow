'''
Descripttion:
version:
Author: Heng Tyrion Wang
Date: 2022-03-03 12:04:03
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:07:01
'''

import tensorflow as tf
from Layers.CrossNet import CrossNet
from Layers.DeepNet import DeepNet
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.FeatureLayer import FeatureLayer

class DeepCross(tf.keras.Model):
    def __init__(self, options, name="DeepCross"):
        super(DeepCross, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.drop_rate = options.dropout_rate
        self.l2_reg = options.l2_reg
        self.num_cross = options.num_cross
        self.hidden_units = eval(options.hidden_units)

    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)
        self.deep_section = DeepNet(self.hidden_units,  self.drop_rate, None)
        self.cross_section = CrossNet(self.num_cross, self.field_dim*self.embedding_dim)
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.dense = tf.keras.layers.Dense(1)


    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        model_input = tf.reshape(embeddings, [-1, self.field_dim*self.embedding_dim])

        deep_term = self.deep_section(model_input, training=training)
        cross_term = self.cross_section(model_input)

        combine_term = tf.concat([deep_term, cross_term], axis =1)

        output = self.dense(combine_term)


        predictions = tf.math.sigmoid(output)

        return predictions
