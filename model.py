import os
import numpy as np
import tensorflow as tf
from modules import *

#np.set_printoptions(threshold=np.inf) 

class Model_PIMIRec(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, time_span, seq_len = 20):
        self.batch_size = batch_size
        self.num_items = n_mid
        self.neg_num = 10
        self.dim = embedding_dim
        self.is_training = tf.placeholder(tf.bool, shape = ())
        self.uid_batch = tf.placeholder(tf.int32, [None, ], name = 'user_id_batch')
        self.itemid_batch = tf.placeholder(tf.int32, [None, ], name = 'target_item_id_batch')
        self.his_itemid_batch = tf.placeholder(tf.int32, [None, seq_len], name = 'his_item_id_batch')
        self.mask = tf.placeholder(tf.float32, [None, seq_len], name = 'his_mask_batch')
        self.adj_matrix = tf.placeholder(tf.float32, [None, seq_len, seq_len + 2], name = 'item_adjacent_batch')
        self.time_matrix = tf.placeholder(tf.int32, [None, seq_len, seq_len], name = 'item_time_interval_batch')
        self.lr = tf.placeholder(tf.float64, [], name = 'learning_rate')

        mask = tf.expand_dims(self.mask, -1)

        with tf.variable_scope("item_embedding", reuse=None):
            self.item_id_embeddings_var = tf.get_variable("item_id_embedding_var", [self.num_items, self.dim], trainable = True)
            self.item_id_embeddings_bias = tf.get_variable("bias_lookup_table", [self.num_items], initializer = tf.zeros_initializer(), trainable = False)
            self.item_eb = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.itemid_batch)
            self.his_itemid_batch_embedded = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.his_itemid_batch)
            
            item_list_emb = tf.reshape(self.his_itemid_batch_embedded, [-1, seq_len, self.dim])
            item_list_emb *= mask

            absolute_pos = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(item_list_emb)[1]), 0), [tf.shape(item_list_emb)[0], 1]),
                vocab_size = seq_len, num_units = hidden_size, scope = "abs_pos", reuse = None)

            item_list_add_pos = item_list_emb + absolute_pos
            item_list_add_pos *= mask

            time_matrix_emb = embedding(self.time_matrix, vocab_size = time_span + 1, num_units = hidden_size, scope = "time_matrix", reuse = None)
            time_W = tf.get_variable("time_W_var", [hidden_size, 1], trainable=True)
            time_matrix_attention = tf.reshape(tf.matmul(tf.reshape(time_matrix_emb, [-1, hidden_size]), time_W), [-1, seq_len, seq_len, 1])

            time_mask = tf.expand_dims(tf.tile(tf.expand_dims(self.mask, axis = 1), [1, seq_len, 1]), axis = -1)
            time_paddings = tf.ones_like(time_mask) * (-2 ** 32 + 1)

            time_matrix_attention = tf.where(tf.equal(time_mask, 0), time_paddings, time_matrix_attention)

            time_matrix_attention = tf.nn.softmax(time_matrix_attention, axis=-2)
            time_matrix_attention = tf.transpose(time_matrix_attention, [0, 1, 3, 2])
            time_emb = tf.squeeze(tf.matmul(time_matrix_attention, time_matrix_emb), axis=2)

            item_list_add_pos_time = item_list_add_pos + time_emb

            item_list_add_pos_time *= mask

        with tf.variable_scope("gcn_interaction", reuse=None):
            node_hidden = item_list_add_pos_time
            b, l, d = get_shape(node_hidden)[0], get_shape(node_hidden)[1], get_shape(node_hidden)[2]
            
            center_node_hidden = tf.reduce_sum(node_hidden, axis=1, keep_dims=False)
            center_node_mask = tf.reduce_sum(self.mask, axis=1, keep_dims=True)
            center_node_hidden /= center_node_mask

            adj_emb = tf.tile(tf.expand_dims(self.adj_matrix, axis=-1), [1, 1, 1, d])
            
            for i in range(3):
                with tf.variable_scope("num_blocks_%d" % i):

                    node_emb = tf.tile(tf.expand_dims(node_hidden, axis=1), [1, l, 1, 1])
                    zeros_tensor = tf.zeros_like(node_emb, dtype=tf.float32)

                    adj_emb_1 = tf.where(tf.equal(adj_emb[:, :, 1:-1, :], 1), node_emb, zeros_tensor)

                    center_node_emb = tf.tile(tf.expand_dims(center_node_hidden, axis=1), [1, l, 1])
                    
                    adj_emb_2 = tf.expand_dims(tf.where(tf.equal(adj_emb[:, :, -1, :], 1), center_node_emb, zeros_tensor[:, :, -1, :]), axis=2)

                    item_node_emb = tf.concat([tf.expand_dims(zeros_tensor[:, :, -1, :], axis=2), adj_emb_1], axis=2)
                    item_node_emb = tf.concat([item_node_emb, adj_emb_2], axis=2)

                    index = tf.where(tf.equal(self.adj_matrix, 1))

                    node_con_emb = tf.reshape(tf.gather_nd(item_node_emb, index), [b, l, -1, d])
                    node_con_emb = tf.concat([node_con_emb, tf.expand_dims(item_list_add_pos, -2)], -2)

                    temp_mask = tf.expand_dims(tf.expand_dims(self.mask, -1), -1)
                    node_con_emb *= temp_mask

                    node_con_emb = tf.reshape(node_con_emb, [b * l, -1, d])
                    node_hidden = tf.reshape(tf.expand_dims(node_hidden, -2), [b * l, -1, d])

                    node_hidden = tf.nn.relu(tf.reshape(tf.squeeze(multihead_attention(node_hidden, node_con_emb, node_con_emb, is_training=self.is_training), axis=-2), [b, l, -1]))

                    node_hidden *= mask
                    center_node_hidden = tf.expand_dims(center_node_hidden, 1)
                    center_con = tf.concat([center_node_hidden, node_hidden], axis=1)
                    center_node_hidden = tf.nn.relu(tf.squeeze(multihead_attention(center_node_hidden, center_con, center_con, is_training=self.is_training), axis=1))

        node_hidden = tf.reshape(node_hidden, [-1, seq_len, self.dim])

        num_heads = num_interest
        with tf.variable_scope("extract_interset", reuse=None):
            item_hidden = tf.layers.dense(node_hidden, hidden_size * 4, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout)

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.item_id_embeddings_var, self.item_id_embeddings_bias, tf.reshape(self.itemid_batch, [-1, 1]), user_emb, self.neg_num * self.batch_size, self.num_items))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, sess, inps):
        feed_dict = {self.uid_batch: inps[0], self.itemid_batch: inps[1], self.adj_matrix: inps[2], self.time_matrix: inps[3], self.his_itemid_batch: inps[4], self.mask: inps[5], self.lr: inps[6], self.is_training: True}
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.item_id_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={self.adj_matrix: inps[0], self.time_matrix: inps[1], self.his_itemid_batch: inps[2], self.mask: inps[3], self.is_training: False})
        return user_embs
    
    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)