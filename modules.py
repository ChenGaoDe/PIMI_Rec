# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf
import numpy as np


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


def embedding(inputs, vocab_size, num_units, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, num_units])
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    return outputs


def multihead_attention(queries, keys, values, num_heads=4,  dropout_rate=0.1, is_training=True, scope="star_multihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Q = tf.layers.dense(queries, d_model, use_bias=True, activation=None)
        K = tf.layers.dense(keys, d_model, use_bias=True, activation=None)
        V = tf.layers.dense(values, d_model, use_bias=True, activation=None)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        d_k = Q_.get_shape().as_list()[-1]

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        outputs /= d_k ** 0.5

        outputs = tf.nn.softmax(outputs)

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 )

        outputs = tf.layers.dense(outputs, d_model, use_bias=True)
 
    return outputs