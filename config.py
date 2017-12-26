class Config:
    def __init__(self):
        self.emb_dim = 200
        self.corpus_path = 'data/corpus.txt'
        self.model_path = 'data/chinese.model'
        self.raw_model_path = 'data/newsblog.vec'
        self.dl_model_path = 'model/1226-strip-150.model'
        self.max_sent_len = 32
        self.lstm_dim = 150
        self.n_filter = 150
        self.filter_size = 3
        self.dropout = 0.5
        self.l2_rate = 1e-3
        self.lr = 1e-3
        self.n_classes = 20
        self.batch_size = 32
        self.epochs = 200
        self.result_path = 'data/results.txt'

