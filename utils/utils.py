'''
Descripttion:
version:
Author: Heng Tyrion Wang
Date: 2022-03-02 16:27:54
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-02 17:47:59
'''

import tensorflow as tf

def decode(data):

    feature_description = {
        "label" : tf.io.FixedLenFeature([], tf.int64),
        "field_idx" : tf.io.FixedLenFeature([], tf.string),
        "field_sub_idx" : tf.io.FixedLenFeature([], tf.string),
        "feature_idx" : tf.io.FixedLenFeature([], tf.string),
        "feature_vals" : tf.io.FixedLenFeature([], tf.string)
    }
    feature = tf.io.parse_single_example(data, feature_description)
    label = feature["label"]
    field_idx = feature["field_idx"]
    field_sub_idx = feature["field_sub_idx"]
    feature_idx = feature["feature_idx"]
    feature_vals = feature["feature_vals"]
    return (field_idx, field_sub_idx, feature_idx, feature_vals), label
