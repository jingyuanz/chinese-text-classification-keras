#coding=utf-8
from config import Config
config = Config()

with open(config.corpus_path) as f:
    lines = f.readlines()
    new_lines = []
    for line in lines:
        query, label = line.strip().split('\t')
        if label == '密码修改' or label == '修改密码' or label=='业务密码重置' or label == '业务密码修改':
            print 1
            label = '改密码'
        elif label == '套餐信息查询':
            print 2
            label = '套餐查询'
        elif label == '家庭礼包咨询':
            print 3
            label = '家庭礼包'
        elif label == '更改套餐':
            print 4
            label = '套餐变更'
        elif label == '积分兑换':
            label = '兑换积分'
            print 5
        elif label == '挂失停机':
            label = '挂失'
        elif label == '月度积分查询':
            label = '积分查询'
        elif label == 'ITV障碍报修' or label == '固定电话障碍' or label == '宽带故障报修':
            label = '故障报修'
        elif label in ['话费查询', '账单查询']:
            label = '查话费'
        elif label in ['积分查询', '月度积分查询', '查询积分']:
            label = '查积分'
        elif label in ['宽带密码修改', '宽带密码重置']:
            label = '宽带密码修改'
        new_line = '\t'.join([query, label]) + '\n'
        new_lines.append(new_line)

with open(config.corpus_path, 'w') as f:
    f.writelines(new_lines)
