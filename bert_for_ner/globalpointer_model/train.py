#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
import os
import json
import glob
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
from global_pointer_model import build_model,build_multi_cls_model
import sys
# root_path = os.path.abspath('..')
# print(root_path)
# sys.path.append(root_path)
from metric_utils import *

seed = 20233
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

maxlen = 128
epochs = 5
batch_size = 16
learning_rate = 2e-5

categories = set()
ckpt_save_path = './checkpoint/best_gp_model.weights'

# bert配置
config_path = 'E:/bert_weight_files/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/bert_weight_files/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/bert_weight_files/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

#训练数据路径
train_path = 'E:/工作空间/中文NER数据/训练数据-供选手下载/CMeEE/CMeEE_train.json'
dev_path = 'E:/工作空间/中文NER数据/训练数据-供选手下载/CMeEE/CMeEE_dev.json'
test_path = 'E:/工作空间/中文NER数据/训练数据-供选手下载/CMeEE/CMeEE_test.json'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def load_data(path):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    text_length = []
    with open(path,'r',encoding='utf8') as f:
        data = json.load(f)
    for d in data:
        text = d['text']
        text_length.append(len(text))
        D.append([text])
        for item in d['entities']:
            D[-1].append((item['start_idx'],item['end_idx'],item['type']))
            categories.add(item['type'])
    df = pd.DataFrame()
    df['len'] = text_length
    print(df['len'].describe())
    return D

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def recognize(text, threshold=0.):
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    scores = model.predict([token_ids, segment_ids])[0]
    scores[:, [0, -1]] -= np.inf #排除[CLS],[SEP]
    scores[:, :, [0, -1]] -= np.inf #
    entities = []
    for l, start, end in zip(*np.where(scores > threshold)):
        entities.append(
            (mapping[start][0], mapping[end][-1], categories[l])
        )
    return sorted(entities,key=lambda d:(d[0],d[1]))

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(dev_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            print('f1 increased from %f to %f' % (self.best_val_f1,f1))
            self.best_val_f1 = f1
            model.save_weights(ckpt_save_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


train_data = load_data(train_path)
random.shuffle(train_data)
dev_data = train_data[:1000]
train_data = train_data[1000:]
test_data = load_data(dev_path)

categories = list(sorted(categories))
model = build_model(config_path,checkpoint_path,len(categories),learning_rate)


def train():
    print(train_data[:2])
    print('训练样本数量',len(train_data))
    print('测试样本数量',len(test_data))
    print(categories)
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(dev_data, batch_size*6)
    evaluator = Evaluator()
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='global_pointer_f1_score', 
        factor=0.5, 
        patience=2, 
        verbose=1,
        mode='max')

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[evaluator,reduce_lr]
    )

    # 评估
    model.load_weights(ckpt_save_path)
    true_laebl,pred_label = [],[]
    for i,d in enumerate(test_data):
        p = [(e[2],e[0],e[1]) for e in recognize(d[0])]
        t = [(e[2],e[0],e[1]) for e in d[1:]]

        true_laebl.extend(t)
        pred_label.extend(p)
    print(classification_report(true_laebl,pred_label,digits=4))


if __name__ == '__main__':
    train()
else:
    model.load_weights(ckpt_save_path)