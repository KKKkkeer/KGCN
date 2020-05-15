import tensorflow as tf
from abc import abstractmethod
import sys

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name, n_iter):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
        self.n_iter = n_iter

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop):
        # score = g(W_r * u, r)
        avg = False
        if not avg:
            n_neighbor = neighbor_vectors.shape[2].value
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, 1, self.dim])
            user_embeddings = tf.tile(user_embeddings, [1, n_neighbor**hop, n_neighbor, 1, 1])
            trans_M = tf.reshape(trans_M, [self.batch_size, n_neighbor**hop, n_neighbor, self.dim, self.dim])
            user_emb_relation = tf.matmul(user_embeddings, trans_M)
            user_emb_relation = tf.squeeze(user_emb_relation, 3)

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_emb_relation * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

    def _mix_neighbor_vectors_KGAT(self, self_vectors, neighbor_vectors, neighbor_relations, trans_M, hop):
        # score = (W_r * e_t)^T * tanh(W_r * e_h + e_r)
        n_neighbor = neighbor_vectors.shape[2].value
        # [batch_size, n_neighbor^hop, 1, dim, 1]
        self_vectors = tf.reshape(self_vectors, [self.batch_size, n_neighbor**hop, 1, self.dim, 1])
        # [batch_size, n_neighbor^hop, n_neighbor, dim, 1]
        self_vectors = tf.tile(self_vectors, [1, 1, n_neighbor, 1, 1])
        # [batch_size, n_neighbor^hop, n_neighbor, dim, dim]
        trans_M = tf.reshape(trans_M, [self.batch_size, n_neighbor**hop, n_neighbor, self.dim, self.dim])
        # [batch_size, n_neighbor^hop, n_neighbor, dim, 1]
        neighbor_relations = tf.expand_dims(neighbor_relations, axis=-1)
        vector1 = tf.nn.tanh(tf.matmul(trans_M, self_vectors) + neighbor_relations)
        # [batch_size, n_neighbor^hop, n_neighbor, dim, 1]
        neighbor_vectors = tf.expand_dims(neighbor_vectors, -1)
        # [batch_size, n_neighbor^hop, n_neighbor, 1, dim]
        vector2 = tf.transpose(tf.matmul(trans_M, neighbor_vectors), [0, 1, 2, 4, 3])
        scores = tf.reshape(tf.matmul(vector2, vector1), [self.batch_size, n_neighbor**hop, n_neighbor]) # [batch_size, n_neighbor^hop, n_neighbor]
        scores_normalized = tf.nn.softmax(scores, dim=-1)
        # [batch_size, n_neighbor^hop, n_neighbor, 1]
        scores_normalized = tf.expand_dims(scores_normalized, axis=-1)
        neighbor_vectors = tf.reshape(neighbor_vectors, [self.batch_size, n_neighbor**hop, n_neighbor, self.dim])
        # [batch_size, -1, dim]
        neighbors_aggregated = tf.reduce_mean(scores_normalized * neighbor_vectors, axis=2)

        return neighbors_aggregated

    def _mix_neighbor_vectors_initial(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, n_iter=1):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name, n_iter)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop):
        # [batch_size, -1, dim]
        # neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings, trans_M, hop)
        # neighbors_agg = self._mix_neighbor_vectors_KGAT(self_vectors, neighbor_vectors, neighbor_relations, trans_M, hop)
        neighbors_agg = self._mix_neighbor_vectors_initial(neighbor_vectors, neighbor_relations, user_embeddings)
        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class BiInteractionAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, n_iter=1):
        super(BiInteractionAggregator, self).__init__(batch_size, dim, dropout, act, name, n_iter)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')
            self.weights2 = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights2')
            self.bias2 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias2')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        # BiInteraction
        output2 = tf.reshape(self_vectors * neighbors_agg, [-1, self.dim])
        output2 = tf.nn.dropout(output2, keep_prob=1-self.dropout)
        output2 = tf.matmul(output2, self.weights2) + self.bias2
        output2 = tf.reshape(output2, [self.batch_size, -1, self.dim])

        return self.act(output) + self.act(output2)

class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, n_iter=1):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name, n_iter)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, n_iter=1):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name, n_iter)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)
