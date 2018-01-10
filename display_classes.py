#coding=utf-8
import codecs
from config import Config
from csv import reader
config = Config()
path = config.corpus_path
classes = set()
with open(path, 'rb') as f:
    lines = f.readlines()
    for row in lines:
        row = row.strip().split('\t')
        clas = row[-1]
        classes.add(clas)

for c in classes:
    print c