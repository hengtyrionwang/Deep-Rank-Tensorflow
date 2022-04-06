'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-04-01 08:43:01
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 08:44:30
'''
import tensorflow as tf

class FeatureLayer(tf.keras.layers.Layer):
    def __init__(self, name="FeatureLayer"):
        super(FeatureLayer, self).__init__()

    def call(self, inputs, training=False):
        field_idx, field_sub_idx, feature_idx, feature_vals = inputs
        field_idx = tf.strings.to_number(tf.strings.split(field_idx), out_type=tf.dtypes.int64).to_sparse()
        field_sub_idx = tf.strings.to_number(tf.strings.split(field_sub_idx), out_type=tf.dtypes.int64).to_sparse()
        feature_idx = tf.strings.to_number(tf.strings.split(feature_idx), out_type=tf.dtypes.int64).to_sparse()
        feature_vals = tf.strings.to_number(tf.strings.split(feature_vals), out_type=tf.dtypes.float32).to_sparse()
        return field_idx, field_sub_idx, feature_idx, feature_vals