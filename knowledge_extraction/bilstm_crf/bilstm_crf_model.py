# coding=utf-8
import keras
from crf_layer import CRF

class BiLstmCrfModel(object):
    def __init__(
            self, 
            max_len,  #传入模型中，句子的最大长度
            vocab_size,  #词向量的字典大小
            embedding_dim, #词向量的维度
            lstm_units, #lstm隐藏单元的数量
            class_nums, #标签数量
            embedding_matrix=None #如果有自己预训练的词向量矩阵，可以在这里声明。如果没有，会自动生成一个
        ):
        super(BiLstmCrfModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.class_nums = class_nums
        self.embedding_matrix = embedding_matrix
        if self.embedding_matrix is not None:
            self.vocab_size,self.embedding_dim = self.embedding_matrix.shape

    def build(self): #建立模型
        inputs = keras.layers.Input(
                shape=(self.max_len,), 
                dtype='int32'
            )
        x = keras.layers.Masking(
                mask_value=0
            )(inputs)
        x = keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                trainable=False,
                weights=self.embedding_matrix,
                mask_zero=True
            )(x)
        x = keras.layers.Bidirectional(
                keras.layers.LSTM( #双向LSTM层
                    self.lstm_units, 
                    return_sequences=True #要让每一个字都输出，而不是只输出最后一个字
                )
            )(x)
        x = keras.layers.TimeDistributed(
                keras.layers.Dropout(
                    0.2
                )
            )(x)
        crf = CRF(self.class_nums) #crf层
        outputs = crf(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam', 
            loss=crf.loss_function, 
            metrics=[crf.accuracy]
            )
        print(model.summary())

        return model
        