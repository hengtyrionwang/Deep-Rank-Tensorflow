'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-03 12:46:08
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:06:59
'''
import tensorflow as tf

from Layers.LinearLayer import LinearLayer
from Layers.FeatureLayer import FeatureLayer

class LR(tf.keras.Model):
    def __init__(self, options, name="LR"):
        super(LR, self).__init__(name=name)
        self.feature_dim = options.feature_dim
        self.embedding_dim = options.embedding_dim
        self.field_dim = options.field_dim
        self.field_sub_dim = options.field_sub_dim
        self.l2_reg = options.l2_reg
    
    def build(self, input_shape):
        self.regularizer = tf.keras.regularizers.L2(self.l2_reg)
        self.feature_layer = FeatureLayer()
        self.linear_section = LinearLayer(self.feature_dim, self.regularizer)
       

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = self.feature_layer(inputs)

        linear_term = self.linear_section([feature_idx, feature_vals])

        predictions = tf.math.sigmoid(linear_term)

        return predictions

