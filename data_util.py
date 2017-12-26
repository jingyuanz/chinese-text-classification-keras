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

    def encode_seq(self, seq):
        if not self.w2v_model:
            return np.zeros((self.config.max_sent_len, self.config.emb_dim))
        seq = list(seq)
        embs = [self.w2v_model[word] for word in seq if word and word in self.w2v_model]
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

    def load_training_set(self):
        with open(self.config.corpus_path) as f:
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
        num_class = len(label_set)
        class_dict = self.config.class_dict
        # for i in range(num_class):
        #     l = label_set.pop()
        #     class_dict[l] = i
        # for val, key in class_dict.items():
        #     print val, key
        # import sys
        # sys.exit(0)
        return raw_sents, np.array(data), labels, class_dict

    def prepare_predict_data(self, raw_sent):
        sent = raw_sent.strip()
        tokens = list(jieba.cut(sent, cut_all=False))
        nopunc = self.preprocess_sent(tokens)
        emb = self.encode_seq(nopunc)
        return emb

    def convert_raw_label_to_class(self, labels, cdict):
        return [cdict[label] for label in labels]


    def preprocess_sent(self, sent):
        return [c for c in sent if c not in self.config.punc]

