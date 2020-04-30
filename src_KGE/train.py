import tensorflow as tf
import numpy as np
from model2 import KGCN
import sys
import random as rd
from time import time

def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    n_kg_triple, all_kg_dict = data[9], data[10]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(
        show_topk, train_data, test_data, n_item)

    exist_users, user_dict = analyze_data(train_data)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_batch = train_data.shape[0] // args.batch_size + 1
        for step in range(args.n_epochs):
            t1 = time()
            loss2, kge_loss, reg_loss = 0., 0., 0.
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            for _ in range(n_batch):
                _, loss = model.train(
                    sess,
                    get_train_feed_dict(model, train_data, args.batch_size,
                                        exist_users, user_dict, n_item))
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data,
                                           args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data,
                                         args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data,
                                         args.batch_size)

            print(
                'epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc,
                   test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(sess, model, user_list,
                                              train_record, test_record,
                                              item_set, k_list,
                                              args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')
            '''
            train the KGE method
            '''
            n_A_batch = n_kg_triple // args.batch_size_kg + 1
            for idx in range(n_A_batch):
                A_batch_data = generate_train_A_batch(args, all_kg_dict, n_entity)
                feed_dict = generate_train_A_feed_dict(model, A_batch_data)

                _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                loss2 += batch_loss
                kge_loss += batch_kge_loss
                reg_loss += batch_reg_loss

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    step, time() - t1, loss2, kge_loss, reg_loss)
            print(perf_str)


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list,
                                         size=user_num,
                                         replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {
        model.user_indices: data[start:end, 0],
        model.item_indices: data[start:end, 1],
        model.labels: data[start:end, 2]
    }
    return feed_dict


def analyze_data(data):
    user_dict = dict()
    for i in range(data.shape[0]):
        if data[i, 2] == 1:
            # if data[i, 0] in user_dict:
            #     user_dict[data[i, 0]] = user_dict[data[i, 0]].append(data[i, 1])
            # else:
            #     user_dict[data[i, 0]] = [data[i, 1]]
            user_dict[data[i, 0]] = user_dict.get(data[i, 0], [])
            user_dict[data[i, 0]].append(data[i, 1])
    exist_users = list(user_dict.keys())
    return exist_users, user_dict


def get_train_feed_dict(model, data, batch_size, exist_users, train_user_dict, n_items):
    n_users = len(exist_users)
    if batch_size <= n_users:
        users = rd.sample(exist_users, batch_size)
    else:
        users = [rd.choice(exist_users) for _ in range(batch_size)]

    def sample_pos_items_for_u(u, num):
        pos_items = train_user_dict[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items_for_u(u, num, n_items):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_i_id = np.random.randint(low=0, high=n_items,size=1)[0]

            if neg_i_id not in train_user_dict[u] and neg_i_id not in neg_items:
                neg_items.append(neg_i_id)
        return neg_items

    pos_items, neg_items = [], []
    for u in users:
        pos_items += sample_pos_items_for_u(u, 1)
        neg_items += sample_neg_items_for_u(u, 1, n_items)

    feed_dict = {
        model.user_indices: users,
        model.pos_item_indices: pos_items,
        model.neg_item_indices: neg_items
    }
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(
            sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set,
              k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(
                sess, {
                    model.user_indices: [user] * batch_size,
                    model.item_indices:
                    test_item_list[start:start + batch_size]
                })
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {
                    model.user_indices: [user] * batch_size,
                    model.item_indices:
                    test_item_list[start:] + [test_item_list[-1]] *
                    (batch_size - len(test_item_list) + start)
                })
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(),
                                        key=lambda x: x[1],
                                        reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def generate_train_A_batch(args, all_kg_dict, n_entities):
    exist_heads = all_kg_dict.keys()

    if args.batch_size_kg <= len(exist_heads):
        heads = rd.sample(exist_heads, args.batch_size_kg)
    else:
        heads = [rd.choice(exist_heads) for _ in range(args.batch_size_kg)]

    def sample_pos_triples_for_h(h, num):
        pos_triples = all_kg_dict[h]
        n_pos_triples = len(pos_triples)

        pos_rs, pos_ts = [], []
        while True:
            if len(pos_rs) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

            t = pos_triples[pos_id][0]
            r = pos_triples[pos_id][1]

            if r not in pos_rs and t not in pos_ts:
                pos_rs.append(r)
                pos_ts.append(t)
        return pos_rs, pos_ts

    def sample_neg_triples_for_h(h, r, num):
        neg_ts = []
        while True:
            if len(neg_ts) == num: break

            t = np.random.randint(low=0, high=n_entities, size=1)[0]
            if (t, r) not in all_kg_dict[h] and t not in neg_ts:
                neg_ts.append(t)
        return neg_ts

    pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

    for h in heads:
        pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
        pos_r_batch += pos_rs
        pos_t_batch += pos_ts

        neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
        neg_t_batch += neg_ts

    batch_data = {}
    batch_data['heads'] = heads
    batch_data['relations'] = pos_r_batch
    batch_data['pos_tails'] = pos_t_batch
    batch_data['neg_tails'] = neg_t_batch
    return batch_data


def generate_train_A_feed_dict(model, batch_data):
    feed_dict = {
        model.h: batch_data['heads'],
        model.r: batch_data['relations'],
        model.pos_t: batch_data['pos_tails'],
        model.neg_t: batch_data['neg_tails'],
    }
    return feed_dict