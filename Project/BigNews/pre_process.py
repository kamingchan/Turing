import json


def read_file(data_file):
    data1 = list()
    data2 = list()
    with open(data_file) as file:
        lines = file.readlines()

        line = lines[1]
        line = [float(x) for x in line.replace('\n', '').split(',')]
        line, label = line[:-1], int(line[-1])
        size = len(line)
        cols = [list() for x in range(size)]

        for line in lines[1:]:
            try:
                line = [float(x) for x in line.replace('\n', '').split(',')]
                line, label = line[:-1], int(line[-1])
                data1.append({
                    'line': line,
                    'label': label
                })
            except ValueError:
                line = [float(x) for x in line.replace('\n', '').split(',')[:-1]]
                data2.append(line)

            for index, num in enumerate(line):
                cols[index].append(num)

        for index, col in enumerate(cols):
            max_d = max(col)
            min_d = min(col)
            rag = max_d - min_d
            for el in data1:
                el['line'][index] = (el['line'][index] - min_d) / rag
            for el in data2:
                el[index] = (el[index] - min_d) / rag

    with open('data/train.json', 'wt') as f:
        json.dump(data1, f, indent=2)

    with open('data/test.json', 'wt') as f:
        json.dump(data2, f, indent=2)


read_file('data/train+test.csv')
