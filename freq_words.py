#coding=utf-8
from jieba import cut
from config import Config
from collections import defaultdict
freq_dict = defaultdict(int)
with open(Config().corpus_path) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().replace('\t','')
        tokens = cut(line, cut_all=False)
        for token in tokens:
            freq_dict[token] += 1

results = list(sorted(freq_dict.items(), key=lambda x:x[1], reverse=True))[:50]
for k,v in results:
    print k, v


