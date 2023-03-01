'''
预训练CNN，用来根据试题文本预测可能的知识点
@ Wang Fei, 2020.11.18
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import gensim.models
from bintrees import FastRBTree
from sklearn.metrics import label_ranking_average_precision_score


knowledge_n = 497     # number of knowledge concepts
batch_size = 32
gpu_n = 0
device = torch.device(('cuda:'+str(gpu_n)) if torch.cuda.is_available() else 'cpu')


def prepare_embedding():
    '''
    reindex the tokens in net_knowledge_train.json (staring index = 1)
    then save the trained token vectors to a numpy file
    :return:
    '''
    with open('data/net_knowledge_train.json', encoding='utf8') as i_f:
        exers = json.load(i_f)

    words, word2id = FastRBTree(), FastRBTree()
    word2vec_model = gensim.models.Word2Vec.load('result/word2vec.model')
    word_npy = []
    for exer in exers:
        for word in exer['text']:
            words.insert(word, True)
    word_count = 0
    word_npy.append([0.] * 100)     # index=0 is a zero-vector
    for word in words:
        if word in word2vec_model:
            word_count += 1
            word2id.insert(word, word_count)
            word_npy.append(word2vec_model[word])
        else:
            print('not found: ' + word)
    word_npy = np.array(word_npy)

    with open('data/word2id.FastRBTree', 'wb') as o_f:
        pickle.dump(word2id, o_f)
    with open('data/word_emb_100', 'wb') as o_f:
        pickle.dump(word_npy, o_f)


class NetKnowledge(nn.Module):
    '''
    the CNN network that is trained and used to predict knowledge concepts from exercise texts
    '''
    def __init__(self):
        self.batch_size = batch_size
        self.embedding_len = 100
        self.sequence_len = 600
        self.output_len = knowledge_n
        self.channel_num1, self.channel_num2, self.channel_num3 = 400, 200, 100
        self.kernel_size1, self.kernel_size2, self.kernel_size3 = 3, 4, 5
        self.pool1 = 3
        self.full_in = (self.sequence_len + self.kernel_size1 - 1) // self.pool1 + self.kernel_size2 + self.kernel_size3 - 2
        super(NetKnowledge, self).__init__()

        with open('data/word2id.FastRBTree', 'rb') as i_f:
            word2id = pickle.load(i_f)
        with open('data/word_emb_100', 'rb') as i_f:
            word_emb_npy = pickle.load(i_f)
        self.word_emb = nn.Embedding(len(word2id) + 1, self.embedding_len, padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_emb_npy))
        self.conv1 = nn.Conv1d(self.embedding_len, self.channel_num1, kernel_size=self.kernel_size1, padding=self.kernel_size1-1, stride=1)
        self.conv2 = nn.Conv1d(self.channel_num1, self.channel_num2, kernel_size=self.kernel_size2, padding=self.kernel_size2-1, stride=1)
        self.conv3 = nn.Conv1d(self.channel_num2, self.channel_num3, kernel_size=self.kernel_size3, padding=self.kernel_size3 - 1, stride=1)
        self.full = nn.Linear(self.full_in, self.output_len)

    def forward(self, x):
        x = self.word_emb(x)
        x = torch.transpose(x, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=self.pool1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, self.channel_num3)
        x = torch.transpose(x, 1, 2).view(-1, self.full_in)
        ret = self.full(x)     # 使用的损失函数包含sigmoid，在预测时需在网络外加sigmoid
        return ret


class TrainDataLoader(object):
    def __init__(self):
        self.batch_size = batch_size
        self.knowledge_n = knowledge_n
        self.max_len = 600    # 文本最大长度
        self.ptr = 0

        with open('data/net_knowledge_train.json', encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open('data/word2id.FastRBTree', 'rb') as i_f:
            self.word2id = pickle.load(i_f)

    def next_batch(self):
        x, y = [], []
        next_ptr = min(len(self.data), self.ptr + self.batch_size)
        for i in range(self.ptr, next_ptr):
            exer = self.data[i]
            word_ids = []
            for word in exer['text'][:self.max_len]:      # 固定长度为self.max_len
                word_id = self.word2id.get(word)
                if word_id is not None:
                    word_ids.append(word_id)
            if len(word_ids) < self.max_len:
                word_ids += [0] * (self.max_len - len(word_ids))    # padding到self.max_len
            x.append(word_ids)
            label = [0.] * self.knowledge_n
            for knowledge_code in exer['knowledge_code']:
                label[knowledge_code - 1] = 1.0
            y.append(label)
        self.ptr = next_ptr
        return torch.LongTensor(x), torch.Tensor(y)

    def is_end(self):
        if len(self.data) - self.ptr < 10:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class TestDataLoader(object):
    def __init__(self, test_fpath):
        self.batch_size = 32
        self.ptr = 0
        self.knowledge_n = knowledge_n
        self.sequence_len = 600
        with open(test_fpath, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open('data/word2id.FastRBTree', 'rb') as i_f:
            self.word2id = pickle.load(i_f)

    def next_batch(self):
        '''
        get the data of next batch
        :return: word_ids, knowledge_label, exer_codes, knowledge_codes
        '''
        if self.is_end():
            return None, None, None, None
        next_ptr = min(self.ptr + self.batch_size, len(self.data))
        x, y, exer_codes, knowledge_codes = [], [], [], []
        for i in range(self.ptr, next_ptr):
            exer = self.data[i]
            word_ids = []
            for word in exer['text'][:self.sequence_len]:
                word_id = self.word2id.get(word)
                if word_id is not None:
                    word_ids.append(word_id)
            if len(word_ids) < self.sequence_len:
                word_ids += [0] * (self.sequence_len - len(word_ids))
            x.append(word_ids)
            knowledge_label = [0.] * self.knowledge_n
            for k_id in exer['knowledge_code']:
                knowledge_label[k_id - 1] = 1.0
            y.append(knowledge_label)
            exer_codes.append(exer['exer_id'])
            knowledge_codes.append(exer['knowledge_code'])
        self.ptr = next_ptr
        return torch.LongTensor(x), torch.Tensor(y), exer_codes, knowledge_codes

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


def train(epoch_n=30):
    data_loader = TrainDataLoader()
    net = NetKnowledge()
    net = net.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(epoch_n):
        batch_count, loss_sum = 0, 0
        data_loader.reset()
        running_loss = 0
        while not data_loader.is_end():
            batch_count += 1
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = net.forward(x)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_sum += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0
        save_snapshot(net, 'netknowledge/model_epoch' + str(epoch + 1))


def test(K, test_fpath, epoch_low=0, epoch_high=50):
    netknowledge = NetKnowledge()
    data_loader = TestDataLoader(test_fpath)
    for epoch in range(epoch_low, epoch_high):
        load_snapshot(netknowledge, 'netknowledge/model_epoch' + str(epoch))
        netknowledge.to(device)
        netknowledge.eval()
        data_loader.reset()
        y_pred_all, y_label_all = [], []
        while not data_loader.is_end():
            x, y, _, _ = data_loader.next_batch()
            x = x.to(device)
            pred = F.sigmoid(netknowledge.forward(x))
            y_label_all += y.tolist()
            y_pred_all += pred.to(torch.device('cpu')).tolist()

        y_pred_all = np.array(y_pred_all)
        y_label_all = np.array(y_label_all)
        # compute MAP
        MAP = label_ranking_average_precision_score(y_label_all, y_pred_all)
        # compute recall@10
        avg_recall = 0
        # select the knowledge concepts with highest K predictions, and then calculate the recall
        for sample_i in range(len(y_pred_all)):
            y_pred = y_pred_all[sample_i]
            y_label = y_label_all[sample_i]
            sort_arg = np.argsort(y_pred)
            recall = 0
            for i in range(K):
                if y_label[sort_arg[-i]] == 1:
                    recall += 1
            recall /= y_label.sum()
            avg_recall += recall
        avg_recall /= len(y_pred_all)

        print('epoch= %d, MAP= %f, avg_recall@%d= %f' % (epoch, MAP, K, avg_recall))
        with open('result/netknowledge_test.txt', 'a', encoding='utf8') as o_f:
            o_f.write('epoch= %d, MAP= %f, avg_recall@%d= %f\n' % (epoch, MAP, K, avg_recall))


def extract_topk(K, epoch, test_fpath, dst_path):
    '''
    predict the topK knowledge concepts for each exercise in test_fpath (data file)
    save the result to dst_path
    对net_knowledge_pred.json中的每一试题预测知识点，取top K
    将标注的知识点与预测的知识点中非标注的构成知识点对，存储
    :param K: 根据netknowledge模型测试的avg_recall@ K 确定
    :param epoch: 用来预测的CNN模型epoch
    :return:
    '''
    netknowledge = NetKnowledge()
    load_snapshot(netknowledge, 'netknowledge/model_epoch' + str(epoch))
    netknowledge.to(device)
    data_loader = TestDataLoader(test_fpath)

    exer_knowledge_pairs = FastRBTree()
    while not data_loader.is_end():
        x, y, exer_codes, knowledge_codes = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        knowledge_pred = netknowledge.forward(x)
        _, topk_indices = torch.topk(knowledge_pred, K, dim=1)
        topk_indices += 1     # 下标从0开始，知识点标号从1开始
        for i in range(len(exer_codes)):
            kn_tags = knowledge_codes[i]
            kn_topks = []
            for kn_topk in topk_indices[i]:
                if kn_topk.item() not in kn_tags:
                    kn_topks.append(kn_topk.item())
            knowledge_pairs = (kn_tags, kn_topks)
            exer_knowledge_pairs.insert(exer_codes[i], knowledge_pairs)

    with open(dst_path, 'wb') as o_f:
        pickle.dump(exer_knowledge_pairs, o_f)


if __name__ == '__main__':
    plt.switch_backend('agg')
    prepare_embedding()
    train(epoch_n=30)
    test(K=20, test_fpath='data/net_knowledge_pred.json')
    extract_topk(K=20, epoch=25, test_fpath='data/net_knowledge_pred.json', dst_path='data/netknowledge_pred_topk_knowledge_pairs.FastRBTree')
