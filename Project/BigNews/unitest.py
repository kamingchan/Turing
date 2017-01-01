import json

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


def read_file(data_file):
    _train_list = list()
    with open(data_file) as file:
        data = json.load(file)
        for line in data:
            _train_list.append(Sample(line))
    return _train_list


w = [-5.49752346e-02, -6.70485976e-02, 5.88342951e-02, -2.32308911e-04,
     -7.96643833e-05, -2.61878843e-04, 4.28496718e-02, -4.36856593e-03,
     3.03085223e-02, 1.21532149e-02, -5.40041626e-02, 6.82362743e-02,
     1.33035502e-02, -3.89314490e-01, -2.02124843e-01, 3.88685826e-01,
     1.75889870e-01, -3.41682037e-01, 1.23227430e-01, 6.57284284e-03,
     9.47830769e-03, 2.00851456e-03, -1.13398795e-01, -2.78664708e-02,
     3.26849816e-01, 2.47038326e-02, 6.93669544e-02, 2.21768926e-02,
     5.39407147e-02, 3.39683320e-02, -7.45134836e-02, -1.35838649e-01,
     -1.45636885e-01, -7.45495197e-02, 2.06912780e-03, 2.56020778e-01,
     1.17473396e-01, 3.73494175e-01, 1.95265964e-01, -1.57111438e-01,
     -3.06516624e-01, 1.25287347e-01, 8.13243944e-02, 6.32074749e-02,
     4.65974373e-03, 9.72440359e-03, -9.89076771e-03, -2.20935815e-02,
     -6.11446693e-02, -1.28185717e-02, -5.50850547e-02, 5.84350329e-02,
     -6.40939919e-02, -1.59744201e-01, -2.50635938e-02, 9.13956716e-02,
     5.46651133e-02, 1.02728483e-01, 9.08190113e-02]

tl = read_file('data/test.json')

with open('084_1.txt', 'wt') as f:
    for t in tl:
        if sigmoid(t.vector.dot(w)) > 0.3:
            f.writelines('1\n')
        else:
            f.writelines('0\n')
