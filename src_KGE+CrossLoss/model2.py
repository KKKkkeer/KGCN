import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, BiInteractionAggregator
from sklearn.metrics import f1_score, roc_auc_score
import sys
'''
较model.py的更改：
1. aggregate() 
  1.1 使用 layer-aggregation mechanism.
2. _build_train()
  2.2 更改self.l2_loss。
'''


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity,
                 adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_weights(n_user, n_entity, n_relation)
        self._build_model()
        self._build_loss_I()

        """
        *********************************************************
        Compute Knowledge Graph Embeddings via TransR.
        """
        self._build_model_II()
        self._build_loss_II()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.batch_size_kg = args.batch_size_kg
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        elif args.aggregator == 'BiInteraction':
            self.aggregator_class = BiInteractionAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64,
                                           shape=[None],
                                           name='user_indices')

        # eval
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        # KGE
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

    def _build_weights(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim],
            initializer=KGCN.get_initializer(),
            name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim],
            initializer=KGCN.get_initializer(),
            name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim],
            initializer=KGCN.get_initializer(),
            name='relation_emb_matrix')
        self.trans_W = tf.get_variable(
            shape=[n_relation, self.dim, self.dim],
            initializer=KGCN.get_initializer(),
            name='trans_W')
        self.agg_W = tf.get_variable(
            shape=[1, self.n_iter+1, 1],
            initializer=KGCN.get_initializer(),
            name='layer_aggregate_weight'
        )


    def _build_model(self):
        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix,
                                                      self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self._get_aggregator()
        self.item_embeddings = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _build_model_II(self):
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        # self.A_kg_score = self._generate_transE_score(h=self.h, t=self.pos_t, r=self.r)

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings = self.entity_emb_matrix
        embeddings = tf.expand_dims(embeddings, 1)

        # head & tail entity embeddings: batch_size *1 * emb_dim
        h_e = tf.nn.embedding_lookup(embeddings, h)
        pos_t_e = tf.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e = tf.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.relation_emb_matrix, r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.trans_W, r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.dim])
        pos_t_e = tf.reshape(tf.matmul(pos_t_e, trans_M), [-1, self.dim])
        neg_t_e = tf.reshape(tf.matmul(neg_t_e, trans_M), [-1, self.dim])

        # Remove the l2 normalization terms
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)
        # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _generate_transE_score(self, h, t, r):
        embeddings = self.entity_emb_matrix
        embeddings = tf.expand_dims(embeddings, 1)

        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf.nn.embedding_lookup(self.relation_emb_matrix, r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = tf.nn.embedding_lookup(self.trans_W, r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])

        # l2-normalize
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1)
        # t_e = tf.math.l2_normalize(t_e, axis=1)

        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)

        return kg_score

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, self.n_neighbor**(i+1)])
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]),
                [self.batch_size, self.n_neighbor**(i+1)])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def _get_aggregator(self):
        self.aggregators = []
        for i in range(self.n_iter):
            aggregator = self.aggregator_class(self.batch_size,
                                               self.dim,
                                               act=tf.nn.tanh,
                                               n_iter=self.n_iter)
            self.aggregators.append(aggregator)

    def aggregate(self, entities, relations):
        # dimensions of entity_vectors
        # {[?, 1, dim], [batch_size, ?, dim], ...}
        entity_vectors = [
            tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities
        ]
        relation_vectors = [
            tf.nn.embedding_lookup(self.relation_emb_matrix, i)
            for i in relations
        ]
        trans_M = [
            tf.nn.embedding_lookup(self.trans_W, i) for i in relations]
        item_vector = []
        out = tf.reshape(entity_vectors[0], [self.batch_size, self.dim, 1])
        for i in range(self.n_iter):
            aggregator = self.aggregators[i]

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, self.n_neighbor**hop, self.n_neighbor, self.dim]

                # [batch_size, -1, dim]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1],
                                                shape),
                    neighbor_relations=tf.reshape(relation_vectors[hop],
                                                  shape),
                    user_embeddings=self.user_embeddings,
                    trans_M=trans_M[hop], hop=hop)
                entity_vectors_next_iter.append(vector)
            out = tf.concat([out, tf.reshape(entity_vectors_next_iter[0], [self.batch_size, self.dim, 1])], 2)
            entity_vectors = entity_vectors_next_iter
            item_vector.append(entity_vectors[0])
        # out_vector = tf.convert_to_tensor(item_vector,
        #                        axis=-1)  # [batch_size, -1, dim * n_iter]
        out_vector = entity_vectors[0]
        out_vector = tf.reshape(out_vector, [-1, self.dim])
        # 说明：
        # 1. vecotr是已激活的
        # 2. 层聚合机制：out = (e1||e2||...||eL)
        out = tf.matmul(out, tf.tile(self.agg_W, [self.batch_size, 1, 1]))
        out = tf.reshape(out, [self.batch_size, self.dim])

        res = tf.reshape(out_vector, [self.batch_size, self.dim])

        return out

    def _build_loss_I(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix) + \
                tf.nn.l2_loss(self.trans_W) + tf.nn.l2_loss(self.agg_W)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
            if self.aggregator_class == BiInteractionAggregator:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(
                    aggregator.weights2)
        self.reg_loss = self.l2_weight * self.l2_loss
        self.loss = self.base_loss + self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_loss_II(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)

        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))
        # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))

        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e) + \
                      tf.nn.l2_loss(self.trans_W)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.l2_weight * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss, self.base_loss, self.reg_loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized],
                                  feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized],
                        feed_dict)

    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)
