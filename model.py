#coding=utf-8
from config import Config
from data_util import DataUtil
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv2D, GlobalMaxPooling1D, Reshape, TimeDistributed, Conv1D
from keras.layers import LSTM
import keras
from keras.regularizers import l2
import numpy as np
from keras.models import save_model, load_model
from keras import backend as K
from keras import Model


class Classifier:
    def __init__(self, predict=False):
        self.config = Config()
        self.du = DataUtil()
        self.raw_sent, self.data, self.raw_labels, self.class_dict = self.du.load_training_set()
        self.classes = self.du.convert_raw_label_to_class(self.raw_labels, self.class_dict)
        self.labels = keras.utils.to_categorical(self.classes, self.du.config.n_classes)

        if predict:
            self.config.dropout = 0
            self.model = load_model(self.config.dl_model_path)
            self.generate_prediction_results()
        else:
            self.model = self.build_model()
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.RMSprop(decay=3e-5, lr=self.config.lr),
                               metrics=['accuracy'])
            self.train()
            self.evaluate()

    def build_model(self):
        input = Input(shape=(self.config.max_sent_len, self.config.emb_dim))
        conv_output = Conv1D(self.config.n_filter, kernel_size=self.config.filter_size, strides=1, activation="relu")(input)
        lstm_output = LSTM(self.config.lstm_dim, dropout=self.config.dropout)(conv_output)
        out = Dense(self.config.n_classes, activity_regularizer=l2(self.config.l2_rate), activation="softmax")(lstm_output)
        model = Model(inputs=[input], outputs=[out])
        return model

    def train(self):

        check = keras.callbacks.ModelCheckpoint(self.du.config.dl_model_path, monitor='val_loss', verbose=1,
                                                save_best_only=False, save_weights_only=False, mode='auto', period=1)
        self.model.fit(self.data, self.labels,
                  batch_size=self.du.config.batch_size,
                  epochs=self.du.config.epochs,
                  verbose=1,
                  validation_split=0.2, callbacks=[check])

    def predict(self, test_data):
        predictions = self.model.predict(test_data)
        predictions = np.argmax(predictions, axis=-1)
        return predictions

    def evaluate(self):
        self.config.dropout = 0
        predictions = self.predict(self.data)
        comparison = (predictions == self.classes)
        acc = np.mean(comparison)
        print "total accuracy: ", acc

    def generate_prediction_results(self):
        with open(self.config.result_path, 'w') as f:
            predictions = self.model.predict(self.data)
            predictions = np.argmax(predictions, axis=-1)
            inv_map = {v: k for k, v in self.class_dict.iteritems()}
            for i in range(len(predictions)):
                sent = self.raw_sent[i]
                ground_truth = self.raw_labels[i]
                pred = inv_map[predictions[i]]
                res = '\t'.join([sent, ground_truth, pred]) + '\n'
                f.write(res)
        print "file written"


if __name__ == '__main__':
    model = Classifier(predict=False)

