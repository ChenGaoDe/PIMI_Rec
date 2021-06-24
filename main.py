#coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
import numpy as np
import faiss
import tensorflow as tf
from data_iterator import DataIterator
from model import *
from tensorboardX import SummaryWriter
import pickle
#np.set_printoptions(threshold=np.inf) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=70) 
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--time_span', type=int, default=64)

best_metric = 0

def prepare_data(src, matrix, target):
    user_id, item_id = src
    adj_matrix, time_matrix = matrix
    hist_item, hist_mask = target
    return user_id, item_id, adj_matrix, time_matrix, hist_item, hist_mask


def get_exp_name(dataset, batch_size, time_span, embedding_dim, lr, maxlen, save = True):
    extr_name = input('Please input the experiment name: ')
    para_name = '_'.join([dataset, 'b' + str(batch_size), 'ts' + str(time_span), 'd' + str(embedding_dim), 'lr' + str(lr), 'len' + str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def evaluate_full(sess, test_data, model, model_path, batch_size, topN):
    item_embs = model.output_item(sess)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    for src, matrix, tgt in test_data:
        nick_id, item_id, adj_matrix, time_matrix, hist_item, hist_mask = prepare_data(src, matrix, tgt)

        user_embs = model.output_user(sess, [adj_matrix, time_matrix, hist_item, hist_mask])

        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                item_list = list(zip(np.reshape(I[i*ni:(i+1)*ni], -1), np.reshape(D[i*ni:(i+1)*ni], -1)))
                item_list.sort(key=lambda x:x[1], reverse=True)

                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                            
                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no+2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
        
        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total

    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


def train(train_file, valid_file, test_file, item_count, batch_size = 128, maxlen = 20, test_iter = 50, lr = 0.001, max_iter = 100, patience = 20):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'

    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('runs/' + exp_name)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, batch_size, maxlen, args.time_span, train_flag=0)
        valid_data = DataIterator(valid_file, batch_size, maxlen, args.time_span, train_flag=1)
        test_data = DataIterator(test_file, batch_size, maxlen, args.time_span, train_flag=2)

        model = Model_PIMIRec(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.time_span, maxlen)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0

            for src, matrix, tgt in train_data:
                data_iter = prepare_data(src, matrix, tgt)
                loss = model.train(sess, list(data_iter) + [lr])
                
                loss_sum += loss
                iter += 1

                if iter % test_iter == 0:
                    log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)

                    metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 20)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()])

                    metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 50)
                    if metrics != {}:
                        log_str += ', ' + ', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()])

                    print(log_str)
                    print(exp_name)

                    writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                    if metrics != {}:
                        for key, value in metrics.items():
                            writer.add_scalar('eval/' + key, value, iter)
                    
                    if 'recall' in metrics:
                        recall = metrics['recall']
                        global best_metric
                        if recall > best_metric:
                            best_metric = recall
                            model.save(sess, best_model_path)
                            trials = 0
                        else:
                            trials += 1
                            if trials > patience:
                                break

                    loss_sum = 0.0
                    test_time = time.time()
                    print("time interval: %.4f min" % ((test_time - start_time)/60.0))
                    sys.stdout.flush()
                
                if iter >= max_iter * 1000:
                        break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        model.restore(sess, best_model_path)

        metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 20)
        print(', '.join(['valid_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
        metrics = evaluate_full(sess, valid_data, model, best_model_path, batch_size, 50)
        print(', '.join(['valid_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))


        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 20)
        print(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 50)
        print(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(test_file, item_count, batch_size = 128, maxlen = 100, lr = 0.001):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)

    model = Model_PIMIRec(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, args.time_span, maxlen)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 20)
        print(', '.join(['test_20 ' + key + ': %.6f' % value for key, value in metrics.items()]))
        metrics = evaluate_full(sess, test_data, model, best_model_path, batch_size, 50)
        print(', '.join(['test_50 ' + key + ': %.6f' % value for key, value in metrics.items()]))


def output(item_count, batch_size = 128, maxlen = 100, lr = 0.001):
    exp_name = get_exp_name(args.dataset, batch_size, args.time_span, args.embedding_dim, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)

    model = Model_PIMIRec(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, args.time_span)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        item_embs = model.output_item(sess)
        np.save('output/' + exp_name + '_emb.npy', item_embs)


if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 313967
        batch_size = 128
        maxlen = 20
        test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'

    if args.p == 'train':
        train(train_file = train_file, valid_file = valid_file, test_file = test_file, 
              item_count = item_count, batch_size = batch_size,  maxlen = maxlen, 
              test_iter = test_iter, lr = args.learning_rate, max_iter = args.max_iter, patience = args.patience)
    elif args.p == 'test':
        test(test_file = test_file, item_count = item_count, batch_size = batch_size, maxlen = maxlen, lr = args.learning_rate)
    elif args.p == 'output':
        output(item_count = item_count, batch_size = batch_size, maxlen = maxlen, lr = args.learning_rate)
    else:
        print('do nothing...')
