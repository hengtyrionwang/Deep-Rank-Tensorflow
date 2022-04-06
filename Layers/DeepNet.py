'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 16:37:43
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-03-14 17:42:24
'''

import tensorflow as tf

class DeepNet(tf.keras.layers.Layer):
    def __init__(self, units=[256,256], drop_rate=0.5, regularizer= None, name="DeepNet"):
        super(DeepNet, self).__init__()
        self.units = units
        self.drop_rate = drop_rate
        self.regularizer = regularizer

    def build(self, input_shape):
        self.dense = []
        self.bn = []
        self.dropout = []
        for i in range(len(self.units)):
            self.dense.append(tf.keras.layers.Dense(self.units[i], activation="relu", kernel_regularizer = self.regularizer))
            self.bn.append(tf.keras.layers.BatchNormalization())
            self.dropout.append(tf.keras.layers.Dropout(self.drop_rate))

    def call(self, inputs, training=False):
        outputs = inputs
        for i in range(len(self.units)):
            outputs = self.dense[i](outputs)
            outputs = self.bn[i](outputs, training=training)
            if training:
                outputs = self.dropout[i](outputs)
        return outputs