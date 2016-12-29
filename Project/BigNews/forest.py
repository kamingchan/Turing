import json

SIZE = 10000

count = [0 for i in range(15858)]

for i in range(SIZE):
    with open(str(i) + '.json') as f:
        j = json.load(f)
    for index, number in enumerate(j):
        count[index] += number

with open('084_1.txt', 'wt') as f:
    for i in count:
        if i > SIZE / 2:
            f.write('1\n')
        else:
            f.write('0\n')
