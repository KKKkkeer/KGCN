import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, BiInteractionAggregator
from sklearn.metrics import f1_score, roc_auc_score
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
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

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
        self.pos_item_indices = tf.placeholder(dtype=tf.int64,
                                               shape=[None],
                                               name='pos_item_indices')
        self.neg_item_indices = tf.placeholder(dtype=tf.int64,
                                               shape=[None],
                                               name='neg_item_indices')

        # eval
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        # KGE
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim * self.n_iter],
            initializer=KGCN.get_initializer(),
            name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim],
            initializer=KGCN.get_initializer(),
            name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim * self.n_iter],
            initializer=KGCN.get_initializer(),
            name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix,
                                                      self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        pos_entities, pos_relations = self.get_neighbors(self.pos_item_indices)
        neg_entities, neg_relations = self.get_neighbors(self.neg_item_indices)

        # [batch_size, dim]
        self._get_aggregator()
        self.pos_item_embeddings = self.aggregate(pos_entities, pos_relations)
        self.neg_item_embeddings = self.aggregate(neg_entities, neg_relations)


        # [batch_size]
        entities, relations = self.get_neighbors(self.item_indices)
        self.eval_item_embeddings = self.aggregate(entities, relations)
        self.scores = tf.reduce_sum(self.user_embeddings *
                                    self.eval_item_embeddings,
                                    axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]),
                [self.batch_size, -1])
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
        item_vector = []
        for i in range(self.n_iter):
            aggregator = self.aggregators[i]

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                shape2 = [
                    self.batch_size, -1, self.n_neighbor,
                    self.dim * self.n_iter
                ]
                # [batch_size, -1, dim]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1],
                                                shape),
                    neighbor_relations=tf.reshape(relation_vectors[hop],
                                                  shape2),
                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
            item_vector.append(entity_vectors[0])
        out_vector = tf.concat(item_vector,
                               axis=-1)  # [batch_size, -1, dim * n_iter]
        out_vector = tf.reshape(out_vector, [-1, self.dim * self.n_iter])
        # 说明：
        # 1. vecotr是已激活的
        # 2. 层聚合机制：out = (e1||e2||...||eL)

        res = tf.reshape(out_vector, [self.batch_size, self.dim * self.n_iter])

        return res

    def _build_train(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.user_embeddings,
                                               self.pos_item_embeddings),
                                   axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.user_embeddings,
                                               self.neg_item_embeddings),
                                   axis=1)
        self.base_loss = tf.reduce_mean(
            tf.nn.softplus(-(pos_scores - neg_scores)))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
            if self.aggregator_class == BiInteractionAggregator:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(
                    aggregator.weights2)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

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
