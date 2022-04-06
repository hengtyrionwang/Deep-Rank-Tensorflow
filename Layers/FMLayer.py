'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 18:31:40
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 18:42:37
'''

import tensorflow as tf

class FMLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, name="FMLayer"):
        super(FMLayer, self).__init__()
        self.feature_dim = feature_dim

    def call(self, inputs, training=False):
        feature_embedding, feature_idx, feature_vals = inputs


        batch_idx = tf.expand_dims(feature_idx.indices[:,0], 1)
        batch_size = feature_idx.indices[-1, 0] + 1

        feat_ids = tf.concat([batch_idx, tf.expand_dims(feature_idx.values, 1)], 1)

        feature = tf.sparse.SparseTensor(feat_ids, feature_vals.values, dense_shape=[batch_size, self.feature_dim])

        feature_square = tf.sparse.SparseTensor(feat_ids, tf.math.pow(feature_vals.values, 2), dense_shape=[batch_size, self.feature_dim])

        sum_square = tf.math.pow(tf.sparse.sparse_dense_matmul(feature, feature_embedding), 2)
        square_sum = tf.sparse.sparse_dense_matmul(feature_square, tf.math.pow(feature_embedding, 2))

        fm_output = tf.math.multiply(0.5, tf.math.subtract(sum_square,square_sum))
        return fm_output