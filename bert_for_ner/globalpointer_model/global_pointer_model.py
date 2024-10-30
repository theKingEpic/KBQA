#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别
import os
import json
import glob
import numpy as np
import random
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.backend import recompute_grad,align,sequence_masking
from bert4keras.backend import apply_rotary_position_embeddings
from keras import initializers, activations
from bert4keras.layers import GlobalPointer,SinusoidalPositionEmbedding
# from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.optimizers import *

def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0.), K.floatx())
    # return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)
    p = K.sum(y_true * y_pred) / K.sum(y_pred)
    r = K.sum(y_true * y_pred) / K.sum(y_true)
    return 2*p*r/(p+r)

def build_model(config_path,checkpoint_path,categories_num,learning_rate):
    model = build_transformer_model(config_path, checkpoint_path)
    output = GlobalPointer(categories_num, 64,RoPE=True)(model.output) #[bs,heads,maxlen,maxlen]

    model = keras.models.Model(model.input, output)
    # model.summary()
    model.compile(
        loss=global_pointer_crossentropy,
        optimizer=Adam(learning_rate),
        metrics=[global_pointer_f1_score]
    )

    return model

def build_multi_cls_model(config_path,checkpoint_path,categories_num,learning_rate):
    model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            return_keras_model=True,
        )
    output_layer = 'Transformer-%d-FeedForward-Norm'
    cls1 = model.get_layer(output_layer % 10).output
    cls2 = model.get_layer(output_layer % 11).output
    mean_cls = keras.layers.Average()([cls1, cls2])
    output = GlobalPointer(categories_num, 64,RoPE=True)(mean_cls)

    model = keras.models.Model(model.input, output)
    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    optimizer = AdamW(learning_rate=learning_rate, weight_decay_rate=0.01)
    # AdamWLR = extend_with_piecewise_linear_lr(AdamW, 'AdamWLR')
    # optimizer = AdamWLR(learning_rate=learning_rate,
    #                     weight_decay_rate=0.01,
    #                     lr_schedule={1000: 1.})
    # model.summary()
    model.compile(
        loss=global_pointer_crossentropy,
        optimizer=optimizer,
        metrics=[global_pointer_f1_score]
    )
    return model