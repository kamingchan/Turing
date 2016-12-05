from numpy import array, zeros, e, power
from numpy.random import random


def sigmoid(_s):
    return 1 / (1 + power(e, -_s))


class Sample(object):
    length = None

    def __init__(self, _vector, _label):
        vec = [1]
        vec.extend(_vector)
        self.vector = array(vec)
        if Sample.length is None:
            Sample.length = self.vector.size
        else:
            assert Sample.length == self.vector.size
        self.label = int(_label)


class Score(object):
    def __init__(self, _w):
        self.w = _w
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

    def test(self, _test_case):
        if _test_case.label == 1:
            if sigmoid(_test_case.vector.dot(self.w)) > 0.4:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if sigmoid(_test_case.vector.dot(self.w)) > 0.4:
                self.fp += 1
            else:
                self.tn += 1

    @property
    def accuracy(self):
        try:
            return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)
        except ZeroDivisionError:
            return None

    @property
    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return None

    @property
    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return None

    @property
    def f1(self):
        try:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        except TypeError:
            return None


def read_file(data_file, train_rate, selected_cols):
    """

    :param data_file:
    :param train_rate: 0-1, divide train list into train or test
    :param selected_cols: columns selected
    """
    _train_list = list()
    _valid_list = list()
    with open(data_file) as file:
        for line in file.readlines()[1:]:
            line = [float(x) for x in line.replace('\n', '').split(',')]
            _s = Sample([line[x] for x in selected_cols], line[-1])
            if random(1) < train_rate:
                _train_list.append(_s)
            else:
                _valid_list.append(_s)
    return _train_list, _valid_list


def err(_train_list, _w):
    _delta_err = zeros(Sample.length, dtype='float64')
    for i in range(Sample.length):
        for ele in _train_list:
            _delta_err[i] += (sigmoid(ele.vector.dot(_w)) - ele.label) * ele.vector[i]
    return _delta_err


if __name__ == '__main__':
    train_list, valid_list = read_file('data/train.csv', 0.8, range(36, 58))
    w = zeros(Sample.length, dtype='float64')
    count = 0
    eta = 0.0001
    last_f1 = 0
    best_f1 = 0
    best_w = None
    while count < 30000000:
        w -= eta * err(train_list, w)
        count += 1
        if count % 2 == 0:
            print('%d times' % count)
            print('Current eta:', eta)
            print('w vector is: %s' % w)
            s = Score(w)
            for v in valid_list:
                s.test(v)
            print('Accuracy:', s.accuracy)
            if s.f1 >= last_f1:
                eta *= 1.00001
            else:
                eta *= 0.999
            if s.f1 > best_f1:
                best_w = w.copy()
                best_f1 = s.f1
            last_f1 = s.f1
            print('Best F1:', best_f1)
            print('F1:', s.f1)
            print('\n')
