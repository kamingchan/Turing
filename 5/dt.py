from math import log, inf


class Sample(object):
    length = None

    def __init__(self, _vector, _label):
        self.vector = _vector
        if Sample.length is None:
            Sample.length = len(self.vector)
        else:
            assert Sample.length == len(self.vector)
        self.label = _label


class Node(object):
    def __init__(self, _id=None, _remain_id=None, _samples=None):
        self.id = _id
        self.remain_id = _remain_id
        self.samples = _samples
        self.sub_nodes = dict()


def read_file(_data_file):
    with open(_data_file) as file:
        for line in file.readlines():
            line = [int(x) for x in line.replace('\n', '').split(',')]
            yield line[:-1], line[-1]


def entropy(_train_list):
    p_0 = len(list(filter(lambda x: x.label == 0, _train_list))) / len(_train_list)
    p_1 = len(list(filter(lambda x: x.label == 1, _train_list))) / len(_train_list)
    return -p_0 * log(p_0, 2) - p_1 * log(p_1, 2)


def id3(_id, _value, _train_list):
    try:
        e = entropy(_train_list)
    except ValueError:
        return 0
    try:
        e_v = entropy(list(filter(lambda x: x.vector[_id] == _value, _train_list)))
    except ValueError:
        return inf
    return e - e_v


def attributes(_samples, _id):
    s = set()
    for sample in _samples:
        s.add(sample.vector[_id])
    return s


if __name__ == '__main__':
    train_list = list()
    for vector, label in read_file('data/train.csv'):
        train_list.append(Sample(vector, label))
    remain_id = list(range(Sample.length))
    # Init root
    tree = Node(_remain_id=remain_id.copy(), _samples=train_list.copy())
    # Begin
    wait_node = [tree]
    while len(wait_node) != 0:
        current_node = wait_node.pop()
        remain_id = current_node.remain_id.copy()
        samples = current_node.samples.copy()
        max_enhance = 0
        best_id = None
        for attribute_id in remain_id:
            total_enhance = sum([id3(attribute_id, x, samples) for x in attributes(samples, attribute_id)])
            if total_enhance > max_enhance:
                best_id = attribute_id
                max_enhance = total_enhance
        current_node.id = best_id
        remain_id.remove(current_node.id)
        for value in attributes(samples, current_node.id):
            node = Node()
            node.remain_id = remain_id.copy()
            node.samples = list(filter(lambda x: x.vector[current_node.id] == value, samples))
            current_node.sub_nodes[value] = node
            if id3(current_node.id, value, samples) != inf:
                wait_node.append(node)
            else:
                node.label = samples[0].label
    # Test
    test_list = list()
    for vector, label in read_file('data/train.csv'):
        test_list.append(Sample(vector, label))
    r = 0
    w = 0
    for test in test_list:
        current_node = tree
        while current_node.id is not None:
            attribute_id = current_node.id
            value = test.vector[attribute_id]
            current_node = current_node.sub_nodes[value]
        if current_node.label == test.label:
            print('HHH')
            r += 1
        else:
            print('ToT')
            w += 1
    print(r, w, r / (r + w))
