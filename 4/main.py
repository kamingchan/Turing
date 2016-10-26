from numpy import array, zeros, sign


class Sample(object):
    length = None

    def __init__(self, _vector, _label):
        vec = [1]
        vec.extend(_vector)
        self.vector = array(vec)
        if Sample.length is None:
            Sample.length = self.vector.size
        assert Sample.length == self.vector.size
        self.label = _label


class Score(object):
    def __init__(self, _w):
        self.w = _w
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def test(self, _test_case):
        if _test_case.label == 1:
            if sign(_test_case.vector.dot(w)) == 1:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if sign(_test_case.vector.dot(w)) == 1:
                self.fp += 1
            else:
                self.tn += 1

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def f1(self):
        return 2 * self.precision + self.recall / (self.precision + self.recall)


def read_file(data_file, label_file):
    with open(data_file) as data_f, open(label_file) as label_f:
        for data_l, label_l in zip(data_f.readlines(), label_f.readlines()):
            yield [int(x) for x in data_l.replace(' \n', '').split(' ')], int(label_l)


def learning(_train_list, _w, find_max=False):
    _max = 0
    _inc = 0
    for ele in _train_list:
        if sign(ele.vector.dot(_w)) != ele.label and abs(ele.vector.dot(_w) - ele.label) > _max:
            _max = abs(ele.vector.dot(_w) - ele.label)
            _inc = ele.vector * ele.label
            if find_max is False:
                break
    if _max == 0:
        return True
    else:
        _w += _inc
        return False


if __name__ == '__main__':
    train_list = list()
    for vector, label in read_file('data/train_data.txt', 'data/train_labels.txt'):
        train_list.append(Sample(vector, label))
    w = zeros(Sample.length, dtype='float64')
    cnt = 0
    while learning(train_list, w, find_max=True) is False:
        cnt += 1
        print('%d times, w vector is: %s' % (cnt, w))
    s = Score(w)
    for vector, label in read_file('data/test_data.txt', 'data/test_labels.txt'):
        s.test(Sample(vector, label))
    print('Accuracy: ', s.accuracy)
    print('Recall: ', s.recall)
    print('Precision: ', s.precision)
    print('F1: ', s.f1)
