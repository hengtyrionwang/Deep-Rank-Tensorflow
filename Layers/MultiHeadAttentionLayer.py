'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 20:20:39
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 20:30:33
'''

import tensorflow as tf

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, num_heads, drop_rate=0.5, name="MultiHeadAttentionLayer"):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.Q_dense=tf.keras.layers.Dense(self.num_units, activation="relu")
        self.K_dense=tf.keras.layers.Dense(self.num_units, activation="relu")
        self.V_dense=tf.keras.layers.Dense(self.num_units, activation="relu")
        self.res_dense=tf.keras.layers.Dense(self.num_units, activation="relu")

        self.bn = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        Q = self.Q_dense(inputs)
        K = self.K_dense(inputs)
        V = self.V_dense(inputs)

        V_res = self.res_dense(inputs)
        
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)

        weights = tf.linalg.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        weights = weights / (K_.get_shape().as_list()[-1]**0.5)

        weights = tf.nn.softmax(weights)

        if training:
            weights = tf.nn.dropout(weights, self.drop_rate)

        outputs = tf.linalg.matmul(weights, V_)

        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis =2)

        outputs += V_res

        outputs = tf.nn.relu(outputs)

        outputs = self.bn(outputs, training=training)

        return outputs