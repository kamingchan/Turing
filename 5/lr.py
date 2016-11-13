from numpy import array, zeros, e


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
            if sigmoid(_test_case.vector.dot(self.w)) > 0.5:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if sigmoid(_test_case.vector.dot(self.w)) > 0.5:
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


def read_file(data_file):
    with open(data_file) as file:
        for line in file.readlines():
            line = [int(x) for x in line.replace('\n', '').split(',')]
            yield line[:-2], line[-1]


def err(_train_list, _w):
    _delta_err = zeros(Sample.length, dtype='float64')
    for i in range(Sample.length):
        for ele in _train_list:
            _delta_err[i] += (sigmoid(ele.vector.dot(_w)) - ele.label) * ele.vector[i]
    return _delta_err


def learning(_train_list, _w, _yita):
    for ele in _train_list:
        if (sigmoid(ele.vector.dot(_w)) >= 0.5) != ele.label:
            _w -= _yita * err(_train_list, _w)
            return False
    return True


if __name__ == '__main__':
    train_list = list()
    for vector, label in read_file('data/train.csv'):
        train_list.append(Sample(vector, label))
    w = zeros(Sample.length, dtype='float64')
    count = 0
    yita = 0.0001
    while learning(train_list, w, yita) is False and count < 10000:
        count += 1
        yita *= 0.9
        print('%d times, w vector is: %s' % (count, w))
    s = Score(w)
    for vector, label in read_file('data/test.csv'):
        s.test(Sample(vector, label))
    print('Accuracy: ', s.accuracy)
    print('Recall: ', s.recall)
    print('Precision: ', s.precision)
    print('F1: ', s.f1)
