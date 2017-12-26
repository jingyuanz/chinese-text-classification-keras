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
from sklearn.utils import shuffle

class Classifier:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil()
        self.inv_map = {v: k for k, v in self.config.class_dict.iteritems()}

        # if predict:
        #     self.config.dropout = 0
        #     self.model = load_model(self.config.dl_model_path)
        #     self.generate_prediction_results()
        # else:

    def run_trainer(self):
        self.raw_sent, self.data, self.raw_labels, _ = self.du.load_training_set()
        self.raw_sent, self.data, self.raw_labels = shuffle(self.raw_sent, self.data, self.raw_labels)
        self.classes = self.du.convert_raw_label_to_class(self.raw_labels, self.config.class_dict)
        self.labels = keras.utils.to_categorical(self.classes, self.du.config.n_classes)
        self.model = self.build_model()
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.RMSprop(lr=self.config.lr),
                           metrics=['accuracy'])
        self.train()
        self.evaluate()

    def run_prediction(self, sentence):
        emb_sent = self.du.prepare_predict_data(sentence)
        emb_sent = emb_sent.reshape([1,self.config.max_sent_len, self.config.emb_dim])
        self.config.dropout = 0
        self.model = load_model(self.config.dl_model_path)
        pred, prob = self.predict(emb_sent)
        pred = pred[0]
        prob = prob[0]
        response = self.inv_map[pred]
        return response, prob

    def build_model(self):
        input = Input(shape=(self.config.max_sent_len, self.config.emb_dim))
        conv_output = Conv1D(self.config.n_filter, kernel_size=self.config.filter_size, strides=1, activation="relu")(input)
        lstm_output = LSTM(self.config.lstm_dim, dropout=self.config.dropout)(conv_output)
        out = Dense(self.config.n_classes, activity_regularizer=l2(self.config.l2_rate), activation="softmax")(lstm_output)
        model = Model(inputs=[input], outputs=[out])
        return model

    def train(self):

        check = keras.callbacks.ModelCheckpoint(self.du.config.dl_model_path, monitor='val_loss', verbose=1,
                                                save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit(self.data, self.labels,
                  batch_size=self.du.config.batch_size,
                  epochs=self.du.config.epochs,
                  verbose=1,
                  validation_split=0.1, callbacks=[check])

    def predict(self, test_data):
        probs = self.model.predict(test_data)
        predictions = np.argmax(probs, axis=-1)
        return predictions, probs

    def evaluate(self):
        self.config.dropout = 0
        predictions = self.predict(self.data)
        comparison = (predictions == self.classes)
        acc = np.mean(comparison)
        print "total accuracy: ", acc

    def generate_prediction_results(self):
        with open(self.config.result_path, 'w') as f:
            predictions, _ = self.model.predict(self.data)
            predictions = np.argmax(predictions, axis=-1)
            for i in range(len(predictions)):
                sent = self.raw_sent[i]
                ground_truth = self.raw_labels[i]
                pred = self.inv_map[predictions[i]]
                res = '\t'.join([sent, ground_truth, pred]) + '\n'
                f.write(res)
        print "file written"


if __name__ == '__main__':
    model = Classifier()
    model.run_trainer()

