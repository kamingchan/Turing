from numpy import array, power, e


class Sample(object):
    length = None

    def __init__(self, _vector):
        vec = [1]
        vec.extend(_vector)
        self.vector = array(vec)
        if Sample.length is None:
            Sample.length = self.vector.size
        else:
            assert Sample.length == self.vector.size


def sigmoid(_s):
    return 1 / (1 + power(e, -_s))


def read_file(data_file, selected_cols):
    """

    :param data_file:
    :param train_rate: 0-1, divide train list into train or test
    :param selected_cols: columns selected
    """
    _train_list = list()
    with open(data_file) as file:
        for line in file.readlines()[1:]:
            line = [float(x) for x in line.replace('\n', '').split(',')[:-1]]
            _s = Sample([line[x] for x in selected_cols])
            _train_list.append(_s)
    return _train_list


w = [-0.06044287, 0.8080602, 0.428194, -0.49540362, -0.64977256, 0.21302874, 0.44351058, 1.59717439, 0.19296422,
     -1.93556644, -0.32231862, -1.12248124, -1.06608499, -0.27378998, -0.73483317, 0.06432835, 0.25406408, -0.27817256,
     -0.09513587, 0.10056402, 0.25595865, 0.53468434, 0.20987778]

tl = read_file('data/test.csv', range(36, 58))

with open('084_1.txt', 'wt') as f:
    for t in tl:
        if sigmoid(t.vector.dot(w)) > 0.5:
            f.writelines('1\n')
        else:
            f.writelines('0\n')
