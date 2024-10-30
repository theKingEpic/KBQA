#! -*- coding: utf-8 -*-
import re
import numpy as np
from tqdm import tqdm
from metrics import *
from train import *


def load_eval_data(data_path,max_len):
    X = []
    y = []
    sentence = []
    labels = []
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    with open(data_path,'r',encoding = 'utf8') as f:
        for line in f.readlines():
            #每行为一个字符和其tag，中间用tab或者空格隔开
            # sentence = [w1,w2,w3,...,wn], labels=[B-xx,I-xxx,,,...,O]
            line = line.strip().split()
            if(not line or len(line) < 2): 
                X.append(sentence)
                y.append(labels)
                sentence = []
                labels = []
                continue
            # word, tag = line[0], line[1].replace('_','-').replace('M','I').replace('E','I').replace('S','B') # BMES -> BIO
            word, tag = line[0], line[1]
            if split_pattern.match(word) and len(sentence)+8 >= max_len:
                sentence.append(word)
                labels.append(tag)
                X.append(sentence)
                y.append(labels)
                sentence = []
                labels = []
            else:
                sentence.append(word)
                labels.append(tag)
    if len(sentence):
        X.append(sentence)
        sentence = []
        y.append(labels)
        labels = []
    return X,y

def predict_label(data,y_true):
    y_pred = []
    for d in data:
        text = ''.join([i[0] for i in d])
        entity_mentions = NER.recognize(text)
        #一开始将标签列表全部初始化为O
        pred = ['O' for _ in range(len(text))]
        b = 0
        for item in entity_mentions:
            word,typ = item[0],item[1]
            start = text.find(word,b)
            end = start + len(word)
            pred[start] = 'B-' + typ
            for i in range(start + 1, end):
                pred[i] = 'I-' + typ
            b += len(word)
        y_pred.append(pred)

    return y_pred

def evaluate():
    #应为验证集路径
    # eval_path = 'C:/Users/wangle/Documents/WeChat Files/w350625640/FileStorage/File/2022-09/eval_data.txt'
    eval_path = './data/cMedQANER/dev2.txt'
    # eval_path = './data/dev.conll'
    test_data,y_true = load_eval_data(eval_path,max_len)
    y_pred = predict_label(test_data,y_true)

    f1 = f1_score(y_true,y_pred,suffix=False)
    p = precision_score(y_true,y_pred,suffix=False)
    r = recall_score(y_true,y_pred,suffix=False)
    acc = accuracy_score(y_true,y_pred)

    print("f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1,p,r,acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))

if __name__ == '__main__':
    evaluate()