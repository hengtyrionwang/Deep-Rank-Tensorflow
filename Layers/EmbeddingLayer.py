'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 18:44:12
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 18:48:41
'''

import tensorflow as tf

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, field_dim, field_sub_dim, name="EmbeddingLayer"):
        super(EmbeddingLayer, self).__init__()
        self.field_dim = field_dim
        self.field_sub_dim = field_sub_dim


    def call(self, inputs, training=False):
        feature_embedding, field_idx, field_sub_idx, feature_idx, feature_vals = inputs

        batch_idx = tf.expand_dims(feature_idx.indices[:,0], 1)
        batch_size = feature_idx.indices[-1, 0] + 1

        field_feat_ids = tf.concat([batch_idx*self.field_dim + tf.expand_dims(field_idx.values, 1), tf.expand_dims(field_sub_idx.values, 1)], 1)

        sparse_index = tf.sparse.SparseTensor(field_feat_ids, feature_idx.values, dense_shape=[batch_size*self.field_dim, self.field_sub_dim])
        sparse_weight = tf.sparse.SparseTensor(field_feat_ids, feature_vals.values, dense_shape=[batch_size*self.field_dim, self.field_sub_dim])

        embeddings = tf.nn.embedding_lookup_sparse(feature_embedding, sparse_index, sparse_weight, combiner="sum")
        return embeddings