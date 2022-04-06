'''
Descripttion: 
version: 
Author: Heng Tyrion Wang
Date: 2022-03-11 20:52:38
LastEditors: Heng Tyrion Wang
Email: hengtyrionwang@gmail.com
LastEditTime: 2022-04-01 09:45:20
'''

import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, field_dim, embedding_dim, hidden_factors, drop_rate, regularizer, name="AttentionLayer"):
        super(AttentionLayer, self).__init__()
        self.field_dim=field_dim
        self.embedding_dim=embedding_dim
        self.drop_rate = drop_rate
        self.regularizer = regularizer
        self.hidden_factors = hidden_factors

    def build(self, input_shape):
        self.weight_attention = self.add_weight('w', shape=[self.hidden_factors[1], self.hidden_factors[0]],initializer="random_normal", regularizer=self.regularizer, trainable=True)
        self.weight_projection = self.add_weight('p', shape=[self.hidden_factors[0]], initializer="random_normal", trainable=True)
        self.weight_bias = self.add_weight('b', shape=[1, self.hidden_factors[0]], initializer="random_normal",  trainable=True)
        self.num_interactions = int(self.field_dim *(self.field_dim-1)/2)

    def call(self, inputs, training=False):
        element_wise_product_list = []

        count = 0

        for i in range(0, self.field_dim):
            for j in range(i+1, self.field_dim):
                element_wise_product_list.append(tf.math.multiply(inputs[:, i, :], inputs[:, j, :]))
                count += 1
   
        element_wise_product = tf.stack(element_wise_product_list)

        element_wise_product = tf.transpose(element_wise_product, perm=[1, 0,  2], name= "element_wise_product")

        attention_mul = tf.reshape(tf.linalg.matmul(tf.reshape(element_wise_product, shape=[-1, self.hidden_factors[1]]), self.weight_attention), shape=[-1, self.num_interactions, self.hidden_factors[0]])
        attention_relu = tf.reduce_sum(tf.math.multiply(self.weight_projection, tf.nn.relu(tf.math.add(attention_mul, self.weight_bias))), 2, keepdims=True)

        attention_out = tf.nn.softmax(attention_relu)

        if training:
            attention_out = tf.nn.dropout(attention_out, self.drop_rate) 
        
        afm = tf.reduce_sum(tf.math.multiply(attention_out, element_wise_product), 1, name="afm")

        if training:
            afm = tf.nn.dropout(afm, self.drop_rate)

        return afm
