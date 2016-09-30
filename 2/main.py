from math import sqrt


class Text(object):
    words_list = list()
    words_set = set()

    def __init__(self, raw_str, split_f):
        """
        :param split_f: function return (id, emotion_id, emotion, words)
        """
        self.id, self.emotion_id, self.emotion, self.words = split_f(raw_str)
        for word in self.words:
            if word not in self.words_set:
                self.words_list.append(word)
                self.words_set.add(word)
        self.onehot = set()

    def update_onehot(self):
        for index, word in enumerate(self.words_list):
            if word in self.words:
                self.onehot.add(index)

    def get_euclidean_distance(self, another_text):
        return sqrt(len(self.onehot | another_text.onehot) - len(self.onehot & another_text.onehot))

    def get_cosine_similiarity(self, another_text):
        return 1 - len(self.onehot & another_text.onehot) / (sqrt(len(self.onehot)) * sqrt(len(another_text.onehot)))

    def get_manhattan_distance(self, another_text):
        return len(self.onehot | another_text.onehot) - len(self.onehot & another_text.onehot)


def get_distance(train_list, test_list):
    result_list = list()
    for test_text in test_list:
        data = dict()
        data['answer'] = test_text.emotion_id

        # Calculate Euclidean distance
        eu_dis = list()
        for train_text in train_list:
            res = dict()
            res['emotion_id'] = train_text.emotion_id
            res['emotion'] = train_text.emotion
            res['dis'] = train_text.get_euclidean_distance(test_text)
            eu_dis.append(res)
        data['eu_dis'] = sorted(eu_dis, key=lambda x: x['dis'])

        # Calculate cosine similiarity
        cs_dis = list()
        for train_text in train_list:
            res = dict()
            res['emotion_id'] = train_text.emotion_id
            res['emotion'] = train_text.emotion
            res['dis'] = train_text.get_cosine_similiarity(test_text)
            cs_dis.append(res)
        data['cs_dis'] = sorted(cs_dis, key=lambda x: x['dis'])

        # Calculate Manhattan distance
        mh_dis = list()
        for train_text in train_list:
            res = dict()
            res['emotion_id'] = train_text.emotion_id
            res['emotion'] = train_text.emotion
            res['dis'] = train_text.get_manhattan_distance(test_text)
            mh_dis.append(res)
        data['mh_dis'] = sorted(mh_dis, key=lambda x: x['dis'])

        # Append
        result_list.append(data)
    return result_list


def knn_classification(k, res_list, weight=False):
    """
    KNN using mode(weight=False) or average(weight=True)
    :param k: k in knn
    :param res_list: list made of dict containing emotion_id and dis
    :param weight:
    :return: emotion_id or 0 meaning unknow
    """
    assert k > 0
    k = min(k, len(res_list))
    k_res_list = res_list[:k]
    while k < len(res_list) and res_list[k - 1] == res_list[k]:
        k_res_list.append(res_list[k])
        k += 1
    statistics = dict()
    for ele in k_res_list:
        em_id = ele['emotion_id']
        if weight:
            point = ele['dis']
        else:
            point = 1
        if em_id not in statistics:
            statistics[em_id] = point
        else:
            statistics[em_id] += point
    sort_statistics = sorted(statistics, key=statistics.get, reverse=True)
    if len(sort_statistics) == 1:
        return sort_statistics
    else:
        first = sort_statistics[0]
        second = sort_statistics[1]
        if statistics[first] > statistics[second]:
            return first
        else:
            return 0


def print_classification(result_list, key_name):
    print(key_name)
    print('%2s  %8s %6s %4s %5s' % ('k', 'succeed', 'unknow', 'fail', 'rate'))
    for k in range(1, 64):
        correct = 0
        unknow = 0
        wrong = 0
        for test_text in result_list:
            ans = test_text['answer']
            ret = knn_classification(k, test_text[key_name])
            if ans == ret:
                correct += 1
            elif ret == 0:
                unknow += 1
            else:
                wrong += 1
        total = correct + unknow + wrong
        print('%2d: %8d %6d %4d %.1f%%' % (k, correct, unknow, wrong, correct * 100 / total))


def do_classification():
    text_list = list()

    def split_1(raw_str):
        """
        :param raw_str: format: [id] [emotion_id] [emotion] [word1, word2, ...]
        """
        str_split = raw_str.replace('\n', '').split(' ')
        return int(str_split[0]), int(str_split[1]), str_split[2], str_split[3:]

    # Read train file
    with open('Classification/train.txt', 'r') as file:
        for line in file.readlines()[1:]:
            new_text = Text(line, split_1)
            text_list.append(new_text)

    # Read test file
    with open('Classification/test.txt', 'r') as file:
        for line in file.readlines()[1:]:
            new_text = Text(line, split_1)
            text_list.append(new_text)

    # Calculate onehot vector of each text
    for text in text_list:
        text.update_onehot()

    train_list = text_list[:246]
    test_list = text_list[246:]

    result_list = get_distance(train_list, test_list)
    print_classification(result_list, 'eu_dis')
    print_classification(result_list, 'cs_dis')
    print_classification(result_list, 'mh_dis')


def knn_regression(k, res_list):
    assert k > 0
    k = min(k, len(res_list))
    k_res_list = res_list[:k]
    while k < len(res_list) and res_list[k - 1] == res_list[k]:
        k_res_list.append(res_list[k])
        k += 1
    statistics = [0] * 6
    for ele in k_res_list:
        dis = ele['dis']
        emotion_list = ele['emotion']
        for index, emotion in enumerate(emotion_list):
            statistics[index] += emotion / dis
    return [x/sum(statistics) for x in statistics]


def print_regression(result_list):
    for k in range(1, 64):
        print(k)
        static_ls  = list()
        for index, test_text in enumerate(result_list):
            static = knn_regression(k, test_text['cs_dis'])
            static_ls.append(static)
        with open('result/%dnn_regression.txt' % k, 'wt') as w_file:
            for static in static_ls:
                w_file.write('%f %f %f %f %f %f\n' % tuple(static))


def do_regression():
    def split2(raw_str):
        """
        :param raw_str: format: [id] [word1, word2, ...] [emotion1, emotion1, ..., emotion6]
        """
        str_split = raw_str.replace('\n', '').split(',')
        words = str_split[1].split(' ')
        emotion_rate = [float(x) for x in str_split[2:]]
        return str_split[0], 0, emotion_rate, words

    train_list = list()
    # Read train file
    with open('Regression/Dataset_train.csv', 'r') as file:
        for line in file.readlines()[1:]:
            new_text = Text(line, split2)
            train_list.append(new_text)

    validation_list = list()
    # Read validation file
    with open('Regression/Dataset_validation.csv', 'r') as file:
        for line in file.readlines()[1:]:
            new_text = Text(line, split2)
            validation_list.append(new_text)

    test_list = list()
    # Read test file
    with open('Regression/Dataset_test.csv', 'r') as file:
        for line in file.readlines()[1:]:
            new_text = Text(line, split2)
            test_list.append(new_text)

    # Calculate onehot vector of each text
    for ele in train_list:
        ele.update_onehot()
    for ele in validation_list:
        ele.update_onehot()
    for ele in test_list:
        ele.update_onehot()

    result_list = get_distance(train_list, validation_list)

    print_regression(result_list)
    # with open('res.json', 'wt') as f:
    #     import json
    #     json.dump(result_list, f, indent=2)


def main():
    # do_classification()
    do_regression()


if __name__ == '__main__':
    main()
