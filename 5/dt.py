from math import log


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
    def __init__(self, _id=None, _remain_id=None, _score=None):
        self.id = _id
        self.remain_id = _remain_id
        self.score = _score
        self.left = Node()
        self.right = Node()


def read_file(_data_file):
    with open(_data_file) as file:
        for line in file.readlines():
            line = [int(x) for x in line.replace('\n', '').split(',')]
            yield line[:-2], line[-1]


def entropy(_train_list):
    p_0 = len(list(filter(lambda x: x.label == 0, _train_list))) / len(_train_list)
    p_1 = len(list(filter(lambda x: x.label == 1, _train_list))) / len(_train_list)
    return -p_0 * log(p_0, 2) - p_1 * log(p_1, 2)


def id3(_id, _train_list):
    pass


if __name__ == '__main__':
    train_list = list()
    for vector, label in read_file('data.train.csv'):
        train_list.append(Sample(vector, label))
    remain_id = range(Sample.length)
    tree = Node(_remain_id=remain_id)
    wait_node = [tree]
    while len(wait_node) != 0:
        current_node = wait_node.pop()
        remain_id = current_node.remain_id
        max_enhance = 0
        max_id = None
        for id in remain_id:
            pass
