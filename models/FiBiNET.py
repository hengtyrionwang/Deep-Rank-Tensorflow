'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-05 12:43:38
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:00:51
'''

import tensorflow as tf
from Layers.DeepNet import DeepNet
from Layers.EmbeddingLayer import EmbeddingLayer
from Layers.LinearLayer import LinearLayer
from Layers.BilinearInteraction import BilinearInteraction
from Layers.SENET import SENET
from Layers.FeatureLayer import FeatureLayer

class FiBiNET(tf.keras.Model):
    def __init__(self, options, name="FiBiNET"):
        super(FiBiNET, self).__init__(name=name)
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
        self.deep_section = DeepNet(self.hidden_units,  self.drop_rate, None)
        self.embedding_section = EmbeddingLayer(self.field_dim, self.field_sub_dim)
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
        self.deep_prediction = tf.keras.layers.Dense(1)
        self.senet = SENET(3, self.field_dim)

        self.bilinear_interaction1 = BilinearInteraction(self.embedding_dim, self.field_dim)
        self.bilinear_interaction2 = BilinearInteraction(self.embedding_dim, self.field_dim)
        self.feature_embedding = self.add_weight('feature_embedding', shape=[self.feature_dim, self.embedding_dim], initializer="random_normal", trainable=True)

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        embeddings = self.embedding_section([self.feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals])

        embeddings = tf.reshape(embeddings, [-1, self.field_dim, self.embedding_dim])

        senet_results = self.senet(embeddings)
        
        senet_interaction = self.bilinear_interaction1(senet_results)

        embedding_interaction = self.bilinear_interaction2(embeddings)

        model_input = tf.concat([senet_interaction, embedding_interaction], axis=-1)

        deep_output = self.deep_section(model_input, training=training)
        deep_term = self.deep_prediction(deep_output)

        combine_term = tf.math.add_n([linear_term, deep_term])

        predictions = tf.math.sigmoid(combine_term)

        return predictions
