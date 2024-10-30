#! -*- coding: utf-8 -*-
import keras
from keras import backend as K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField

class SetLearningRate:
    """
    层的一个包装，用来设置当前层的学习率
    """
    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器
        
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)

def bert_bilstm_crf(config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate):
	bert = build_transformer_model(
			config_path = config_path,
			checkpoint_path = checkpoint_path,
			model = 'bert',
			return_keras_model = False 
			# True，x = bert.output
		)
	x = bert.model.output # [batch_size, seq_length, 768]
	lstm = SetLearningRate(
			keras.layers.Bidirectional(
				keras.layers.LSTM(
					lstm_units,
					kernel_initializer='he_normal',#权重初始化
					return_sequences=True
				)
			), 
			500, 
			True
		)(x) # [batch_size, seq_length, lstm_units * 2]

	x = keras.layers.concatenate( #这里只是把Bert和BiLSTM层的输入进行了拼接，也可以不这么做
			[lstm,x],
			axis=-1
		) # [batch_size, seq_length, lstm_units * 2 + 768]

	x = keras.layers.TimeDistributed( #lstm之后做一个dropout
			keras.layers.Dropout(drop_rate)
		)(x) # [batch_size, seq_length, lstm_units * 2 + 768]

	x = SetLearningRate(
			keras.layers.TimeDistributed(
				keras.layers.Dense( #全连接
						num_labels,
						activation='relu',
						kernel_initializer='he_normal',
					)
			), 
			500, 
			True
		)(x) # [batch_size, seq_length, num_labels]

	crf = ConditionalRandomField(lr_multiplier=500)
	output = crf(x)

	model = keras.models.Model(bert.input, output)
	# model.summary()
	#优化器（optimizer）是一种用于更新模型参数以最小化训练误差的算法
	model.compile(
			loss=crf.sparse_loss,
			optimizer=Adam(leraning_rate),
			metrics=[crf.sparse_accuracy]
		)

	return model,crf



if __name__ == '__main__':
	config_path = 'C:/Users/胥君玥/Desktop/bert_for_ner/bert_weight_files/bert_config.json'
	checkpoint_path = 'C:/Users/胥君玥/Desktop/bert_for_ner/bert_weight_files/bert_model.ckpt' #C:\Users\胥君玥\Desktop\KBQA-py3.6\nlu\bert_intent_recognition\bert_weight_files
	num_labels = 21
	lstm_units = 128
	drop_rate = 0.1
	leraning_rate = 5e-5
	model , crf = bert_bilstm_crf(
		config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate)
	print(model.summary())