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


def entropy(_samples):
    p_0 = len(list(filter(lambda x: x.label == 0, _samples))) / len(_samples)
    p_1 = len(list(filter(lambda x: x.label == 1, _samples))) / len(_samples)
    return -p_0 * log(p_0, 2) - p_1 * log(p_1, 2)


def id3(_id, _value, _samples):
    try:
        e = entropy(_samples)
    except ValueError:
        return 0
    try:
        e_v = entropy(list(filter(lambda x: x.vector[_id] == _value, _samples)))
    except ValueError:
        return inf
    return e - e_v


def attributes(_samples, _id):
    _s = set()
    for _sample in _samples:
        _s.add(_sample.vector[_id])
    return _s


def split_info(_id, _value, _samples):
    _len_total = len(_samples)
    _len_p = len(list(filter(lambda x: x.vector[_id] == _value, _samples)))
    _len_n = _len_total - _len_p
    return -(_len_p / _len_total + log(_len_p / _len_total, 2)) - (_len_n / _len_total + log(_len_n / _len_total, 2))


def c4dot5(_id, _value, _samples):
    return id3(_id, _value, _samples) / split_info(_id, _value, _samples)


def neg_gini(_id, _value, _samples):
    _len_total = len(_samples)
    _len_p = len(list(filter(lambda x: x.vector[_id] == _value, _samples)))
    _len_n = _len_total - _len_p
    _gini = 1 - (_len_p / _len_total) ** 2 - (_len_n / _len_total) ** 2
    return -_gini


if __name__ == '__main__':
    train_list = list()
    for vector, label in read_file('data/train.csv'):
        train_list.append(Sample(vector, label))
    remain_id = list(range(Sample.length))
    # Init root
    tree = Node(_remain_id=remain_id.copy(), _samples=train_list.copy())
    tree.epsilon = list()
    tree.beta = 1
    # Begin
    wait_node = [tree]
    while len(wait_node) != 0:
        current_node = wait_node.pop()
        remain_id = current_node.remain_id.copy()
        samples = current_node.samples.copy()
        max_enhance = -inf
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
            judge = c4dot5(current_node.id, value, samples)
            if judge < inf and len(node.remain_id) > 0:
                print(judge)
                if len(list(filter(lambda x: x.label == 1, samples))) > len(
                        list(filter(lambda x: x.label == 0, samples))):
                    node.label = 1
                else:
                    node.label = 0
                wait_node.append(node)
            else:
                node.pos = 0
                node.neg = 0
                for sample in node.samples:
                    if sample.label == 1:
                        node.pos += 1
                    else:
                        node.neg += 1
                if node.pos > node.neg:
                    node.label = 1
                    tree.epsilon.append(node.neg)
                else:
                    node.label = 0
                    tree.epsilon.append(node.pos)
    # Test
    test_list = list()
    for vector, label in read_file('data/test.csv'):
        test_list.append(Sample(vector, label))
    r = 0
    w = 0
    for test in test_list:
        current_node = tree
        while current_node.id is not None:
            attribute_id = current_node.id
            value = test.vector[attribute_id]
            try:
                current_node = current_node.sub_nodes[value]
            except KeyError:
                break
        if current_node.label == test.label:
            r += 1
        else:
            w += 1
    print('Epsilon:', (sum(tree.epsilon) + len(tree.epsilon) * tree.beta) / len(tree.samples))
    print(r, w, r / (r + w))
