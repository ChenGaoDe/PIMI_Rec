import numpy
import json
import random
import numpy as np

class DataIterator:
    def __init__(self, source, batch_size=128, maxlen=100, time_span=128, train_flag=0):
        self.read(source)
        self.users = list(self.users)
        
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.time_span = time_span
        self.index = 0

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph = {}
        self.users = set()

        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = float(conts[2])
                self.users.add(user_id)

                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append([item_id, time_stamp])

        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])

        for user_id, items in self.graph.items():
            time_list = list(map(lambda x: x[1], items))
            time_min = min(time_list)
            self.graph[user_id] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / 86400.0) + 1)], items))

        self.users = list(self.users)

    def compute_adj_matrix(self, mask_seq, item_num):
        node_num = len(mask_seq)

        adj_matrix = np.zeros([node_num, node_num + 2], dtype=np.int32)

        adj_matrix[0][0] = 1
        adj_matrix[0][1] = 1
        adj_matrix[0][-1] = 1

        adj_matrix[item_num - 1][item_num - 1] = 1
        adj_matrix[item_num - 1][item_num] = 1
        adj_matrix[item_num - 1][-1] = 1
        
        for i in range(1, item_num - 1):
            adj_matrix[i][i] = 1
            adj_matrix[i][i + 1] = 1
            adj_matrix[i][-1] = 1

        if (item_num < node_num):
            for i in range(item_num, node_num):
                adj_matrix[i][0] = 1
                adj_matrix[i][1] = 1
                adj_matrix[i][-1] = 1

        return adj_matrix

    def compute_time_matrix(self, time_seq, item_num):
        time_matrix = np.zeros([self.maxlen, self.maxlen], dtype=np.int32)
        for i in range(item_num):
            for j in range(item_num):
                span = abs(time_seq[i] - time_seq[j])
                if span > self.time_span:
                    time_matrix[i][j] = self.time_span
                else:
                    time_matrix[i][j] = span
        return time_matrix

    def __next__(self):
        if self.train_flag == 0:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        item_id_list = []
        adj_matrix_list = []
        time_matrix_list = []
        hist_item_list = []
        hist_mask_list = []
        for user_id in user_id_list:
            item_list = [x[0] for x in self.graph[user_id]]
            time_list = [x[1] for x in self.graph[user_id]]
            if self.train_flag == 0:
                k = random.choice(range(4, len(item_list)))
                item_id_list.append(item_list[k])
            else:
                k = int(len(item_list) * 0.8)
                item_id_list.append(item_list[k:])
            if k >= self.maxlen:
                hist_item_list.append(item_list[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * self.maxlen, self.maxlen))
                time_matrix_list.append(self.compute_time_matrix(time_list[k - self.maxlen: k], self.maxlen))
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))
                adj_matrix_list.append(self.compute_adj_matrix([1.0] * k + [0.0] * (self.maxlen - k), k))
                time_matrix_list.append(self.compute_time_matrix(time_list[:k] + [0] * (self.maxlen - k), k))
                
        return (user_id_list, item_id_list), (adj_matrix_list, time_matrix_list), (hist_item_list, hist_mask_list)

