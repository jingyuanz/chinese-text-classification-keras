from config import Config
from random import shuffle
with open(Config().corpus_path) as f:
    lines = f.readlines()
    total = len(lines)
    test_portion = total/Config().test_fold
    shuffle(lines)
    test_lines = lines[:test_portion]
    train_lines = lines[test_portion:]

with open(Config().test_corpus_path, 'w') as f:
    f.writelines(test_lines)

with open(Config().train_corpus_path, 'w') as f:
    f.writelines(train_lines)