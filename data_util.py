#coding=utf-8
import pickle
import numpy as np
from config import Config
import codecs
import jieba

class DataUtil:
    def __init__(self):
        self.config = Config()
        self.w2v_model = self.build_w2v_model(True)

    def build_w2v_model(self, load=False):
        if not load:
            model = self.load_w2v_model()
            self.save_model(model)
        else:
            model = self.load_model()
        return model

    # def build_w2v_model(self, load=False):
    #     return None

    def save_model(self, model):
        path = 'data/chinese.model'
        with open(path, 'w') as f:
            pickle.dump(model, f)
        print 'model saved'

    def load_model(self, path='data/chinese.model'):
        with open(path, 'r') as f:
            model = pickle.load(f)
        print "model loaded"
        print model['</s>'].shape
        return model

    def correct_tokens(self, tokens):
        new_tokens = tokens
        for i in range(len(tokens)):
            token = tokens[i]
            if token == u'我想查' or token == u'帮查':
                new_tokens[i] = u'查'
            elif token == u'IPHONE8':
                new_tokens[i] = u'手机'
            elif token == u'日租卡':
                new_tokens[i] = u'日租'
            elif token in [u'我本机', u'帮本机', u'查本机', u'改本机']:
                new_tokens[i] = u'本机'
            elif token == u'他机':
                new_tokens[i] = u'别人'
            elif token == u'我想装':
                new_tokens[i] = u'装'
            elif token == u'积量':
                new_tokens[i] = u'流量'
            elif token == u'办停':
                new_tokens[i] = u'停机'
        assert len(new_tokens) == len(tokens)
        return new_tokens

    def encode_seq(self, seq):
        if not self.w2v_model:
            return np.zeros((self.config.max_sent_len, self.config.emb_dim))
        seq = list(seq)
        seq = self.correct_tokens(seq)
        filtered_seq = [w for w in seq if w and w in self.w2v_model]
        embs = [self.w2v_model[word] for word in filtered_seq]
        diff = set(seq) - set(filtered_seq)
        for w in diff:
            print w
        len_diff = self.config.max_sent_len - len(embs)
        padding = [0.0] * self.config.emb_dim
        embs.extend(len_diff * [padding])
        embs = np.array(embs)
        if embs.shape != (self.config.max_sent_len, self.config.emb_dim):
            print embs.shape
            print len_diff
            print seq
        assert embs.shape == (self.config.max_sent_len, self.config.emb_dim)
        return embs

    def load_w2v_model(self, fname="data/newsblogbbs.vec", load=False):
        print "load model..."
        with codecs.open(fname, 'r', "utf-8") as f:
            vocab = self.load_w2v_data(f)
        return vocab

    def load_w2v_data(self, f):
        size = 0
        # vocab = []
        # feature = []
        vocab_dict = {}
        flag = 0
        while True:
            line = f.readline()
            if not line:
                break
            if flag == 0:
                line = line.strip().split()
                _, size = int(line[0]), int(line[1])
                flag = 1
                continue
            line = line.strip().split()
            if not line:
                continue
            w = line[0]
            vec = [float(i) for i in line[1:]]
            if len(vec) != size:
                continue
            vec = np.array(vec)
            length = np.sqrt((vec ** 2).sum())
            vec /= length
            # print length,vec
            vocab_dict[w] = vec
        return vocab_dict

    def load_data_set(self, type='train'):
        if type == 'test':
            path = self.config.test_corpus_path
        else:
            path = self.config.train_corpus_path
        with open(path) as f:
            lines = f.readlines()
            data = []
            labels = []
            label_set = set()
            raw_sents = []
            for line in lines:
                # print line[-1]
                content = line.strip().split('\t')
                sent, label = content[0], content[1]
                raw_sents.append(sent)
                tokens = jieba.cut(sent, cut_all=False)
                tokens = list(tokens)
                nopunc_tokens = self.preprocess_sent(tokens)
                emb = self.encode_seq(nopunc_tokens)
                # emb = np.reshape(emb, (self.config.max_sent_len, self.config.emb_dim,1))
                # emb = np.reshape(emb, (self.config.emb_dim, 1))
                data.append(emb)
                labels.append(label)
                label_set.add(label)
        return raw_sents, np.array(data), labels



    def prepare_predict_data(self, raw_sent):
        sent = raw_sent.strip()
        tokens = list(jieba.cut(sent, cut_all=False))
        nopunc = self.preprocess_sent(tokens)
        emb = self.encode_seq(nopunc)
        return emb

    def convert_raw_label_to_class(self, labels, cdict):
        # c = 0
        # for label in labels:
        #     c += 1
        #     print cdict[label], c
        return [cdict[label] for label in labels]


    def preprocess_sent(self, sent):
        processed = [c for c in sent if c not in self.config.punc and not c.isdigit() and c not in self.config.freq_words]
        return processed


