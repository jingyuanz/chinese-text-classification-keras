from config import Config

with open(Config().result_path) as f:
    lines = f.readlines()
    lines = [line.strip().split('\t') for line in lines]
    wrong_lines = []
    for line in lines:
        if line[1] != line[2]:
            wrong_lines.append(line)

with open(Config().result_path, 'wb') as f:
    for line in wrong_lines:
        f.write('\t'.join(line) + '\n')
