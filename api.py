#coding=utf-8
from config import Config
from data_util import DataUtil
from model import Classifier
import operator

class API:
    def __init__(self):
        self.model = Classifier()

    def predict_sentence(self, sent):
        res, prob = self.model.run_prediction(sent)
        distrib = {}
        for i in range(len(prob)):
            distrib[self.model.inv_map[i]] = prob[i]

        sorted_distrib = sorted(distrib.items(), key=operator.itemgetter(1), reverse=True)
        for k, v in sorted_distrib:
            print k, v
        print ">> ", res
        print
        return res, sorted_distrib


    def train_corpus(self, corpus_path=None, new_model_save_path=None):
        if corpus_path:
            self.model.config.corpus_path = corpus_path
        if new_model_save_path:
            self.model.config.corpus_path = new_model_save_path
        self.model.train()


if __name__ == '__main__':
    api = API()
    api.predict_sentence(u"我想查话费行不行")
    api.predict_sentence(u"我不想用现在的套餐了")
    #api.train_corpus()
