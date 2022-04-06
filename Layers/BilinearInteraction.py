'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 22:03:03
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-11 22:03:04
'''

import tensorflow as tf

class BilinearInteraction(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, field_dim, name="BilinearInteraction"):
        super(BilinearInteraction, self).__init__()
        self.embedding_dim = embedding_dim
        self.field_dim = field_dim

    def build(self, input_shape):
        self.weight = self.add_weight('w', shape=[self.embedding_dim, self.embedding_dim], initializer="random_normal", trainable=True)

    def call(self, inputs, training=False):
        vidots = []

        for i in range(0, self.field_dim):
            vidots.append(tf.linalg.matmul(inputs[:,i,:], self.weight))
        pdata = []

        for i in range(0, self.field_dim):
            for j in range(0, self.field_dim):
                pdata.append(tf.math.multiply(vidots[i], inputs[:, j, :]))

        output = tf.concat(pdata, axis=1)
        return output