# coding=utf-8
class Config:
    def __init__(self):
        self.emb_dim = 200
        self.corpus_path = 'data/corpus.txt'
        self.xls_corpus_path = 'data/sup_corpus.txt'
        self.train_corpus_path = 'data/train.txt'
        self.test_corpus_path = 'data/test.txt'
        self.model_path = 'data/chinese.model'
        self.raw_model_path = 'data/newsblogbbs.vec'
        self.dl_model_path = 'model/108-300-valloss.model'
        self.final_round_model_path = 'model/108-300-final.model'
        self.max_sent_len = 28
        self.lstm_dim = 300
        self.n_filter = 300
        self.filter_size = 3
        self.dropout = 0.5
        self.l2_rate = 1e-3
        self.lr = 5e-4
        self.n_classes = 26
        self.h1_dim = 100
        self.batch_size = 32
        self.test_fold = 10
        self.epochs = 150
        self.result_path = 'data/results.txt'
        self.punc = u"！？。＂＃$＄％&＆'＇()（）*＊+＋，-－/／:：;；<＜=＝>＞@[［＼\］] \
            ＾^＿_｀`{｛|｜}｝~～《》｟｠｢｣､、〃「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟 \
            〰〾〿–—‘’‛“”„‟…‧﹏."
        self.class_dict = {'兑换积分': 1,
                           '套餐变更': 2,
                           '宽带到期时间查询': 3,
                           '路由器设置不成功': 4,
                           '人工服务': 5,
                           '宽带密码修改': 6,
                           '宽带账号查询': 7,
                           '信用额度查询': 8,
                           '宽带维修进度查询': 9,
                           '装宽带': 10,
                           '装移机进度查询': 11,
                           '翼支付客服电话': 12,
                           '查话费': 13,
                           '优惠活动咨询': 14,
                           '流量包办理': 15,
                           '呼叫转移': 16,
                           '改密码': 17,
                           '流量查询': 18,
                           '故障报修': 19,
                           '复机业务': 20,
                           '国际漫游': 21,
                           '查积分': 22,
                           '套餐查询': 23,
                           '他机信息查询': 24,
                           '家庭礼包': 25,
                           '挂失': 0
                           }
        self.freq_words = [u'我', u'下', u'一下', u'我加', u'我要',u'要',u'的', u'了', u'吗', u'吧', u'呢', u'啊', '我']
        # self.freq_words = []

'''2001：话费查询
2002：套餐查询
2003：流量查询
2004：积分查询
2005：账单查询
2006：切换号码
2007：优惠活动
2008：家庭礼包
2009：故障报修
2010：宽带故障
2011：修改密码
2012：更改套餐
2013：积分兑换
2014：流量包办理
2015：手机挂失
2016：信用额度查询
2017：宽带账号查询
2018：宽带到期时间查询
2019：宽带维修进度查询
2020：装移机进度查询
2021：翼支付客服电话咨询
'''