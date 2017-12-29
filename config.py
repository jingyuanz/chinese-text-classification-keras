# coding=utf-8
class Config:
    def __init__(self):
        self.emb_dim = 200
        self.corpus_path = 'data/corpus.txt'
        self.train_corpus_path = 'data/train.txt'
        self.test_corpus_path = 'data/test.txt'
        self.model_path = 'data/chinese.model'
        self.raw_model_path = 'data/newsblogbbs.vec'
        self.dl_model_path = 'model/1229-300-valloss.model'
        self.final_round_model_path = 'model/1229-300-final.model'
        self.max_sent_len = 28
        self.lstm_dim = 300
        self.n_filter = 300
        self.filter_size = 3
        self.dropout = 0.5
        self.l2_rate = 1e-3
        self.lr = 1e-3
        self.n_classes = 13
        self.h1_dim = 100
        self.batch_size = 32
        self.test_fold = 10
        self.epochs = 200
        self.result_path = 'data/results.txt'
        self.punc = u"！？。＂＃$＄％&＆'＇()（）*＊+＋，-－/／:：;；<＜=＝>＞@[［＼\］] \
            ＾^＿_｀`{｛|｜}｝~～《》｟｠｢｣､、〃「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟 \
            〰〾〿–—‘’‛“”„‟…‧﹏."
        self.class_dict = {'积分查询': 0,
                           '兑换积分': 1,
                           '优惠活动咨询': 2,
                           '流量包办理': 3,
                           '账单查询': 4,
                           '故障报修': 5,
                           '话费查询': 6,
                           '套餐变更': 7,
                           '套餐查询': 8,
                           '流量查询': 9,
                           '家庭礼包': 10,
                           '改密码': 11,
                           '挂失': 12,
                           }
        # self.freq_words = ['我', '的', '我要', '了', '吗', '想', '你', '你们', '吧', '呢', '啊']
        self.freq_words = []