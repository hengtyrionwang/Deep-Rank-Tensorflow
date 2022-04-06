'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 18:54:14
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 19:04:57
'''

import tensorflow as tf

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, regularizer=None, name="LinearLayer"):
        super(LinearLayer, self).__init__()
        self.feature_dim = feature_dim
        self.regularizer = regularizer

    def build(self, input_shape):
        self.weight_linear = self.add_weight('w', shape=[self.feature_dim, 1], initializer="random_normal", regularizer=self.regularizer, trainable=True)
        self.bias = self.add_weight('b', shape=[1], initializer='zeros', trainable=True)

    def call(self, inputs, training=False):
        feature_idx, feature_vals = inputs

        batch_idx = tf.expand_dims(feature_idx.indices[:,0], 1)
        batch_size = feature_idx.indices[-1, 0] + 1

        feat_ids = tf.concat([batch_idx, tf.expand_dims(feature_idx.values, 1)], 1)

        feature = tf.sparse.SparseTensor(feat_ids, feature_vals.values, dense_shape=[batch_size, self.feature_dim])

        linear_term = tf.math.add(tf.sparse.sparse_dense_matmul(feature, self.weight_linear), self.bias)
        return linear_term