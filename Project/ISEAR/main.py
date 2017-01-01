from random import random


class Emotion(object):
    # 所有情感在测试集出现的次数，实则是测试集的大小，封装一下方便统计
    __global_count = 0

    def __init__(self):
        # 统计每个单词在该情感中出现次数
        self.__words_count = dict()
        # 统计不重复单词
        self.__words_set = set()
        # 单词总数，应该等于上面所有 key 对应 value 的 sum
        self.__total_words = 0
        # 这个情感在测试集出现的次数
        self.__self_count = 0

    def add(self, word, n=1):
        """
        表示这个单词在这个情感中出现
        :param word: 单词
        :param n: 出现个数，默认 1
        """
        if word in self.__words_count:
            self.__words_count[word] += n
        else:
            self.__words_count[word] = n
        self.__total_words += n
        self.__words_set.add(word)

    @property
    def probability(self):
        """
        :return: 返回该情感在测试集中出现的概率
        """
        return self.__self_count / Emotion.__global_count

    def appear(self):
        """
        该情感在测试集中出现一次，该函数应该被调用一次
        """
        Emotion.__global_count += 1
        self.__self_count += 1

    def get_word_probability(self, word, laplace_smoothing=False, alpha=1):
        """
        :param alpha: 0-1 之间的参数，取决于 label 的多少
        :param word: 一个测试集单词
        :param laplace_smoothing: 是否开启拉普拉斯平滑
        :return: 返回该单词在该情感中出现的未归一化概率
        """
        if laplace_smoothing is True:
            if word in self.__words_count:
                return (self.__words_count[word] + alpha) / (self.__total_words + len(self.__words_set) * alpha)
            else:
                return alpha / (self.__total_words + len(self.__words_set) * alpha)
        else:
            if word in self.__words_count:
                return self.__words_count[word] / self.__total_words
            else:
                return 0


def read_file(_file_name, _rand=False):
    _stop_words = set()
    with open('data/stop_words.txt') as f:
        for _line in f.readlines():
            _stop_words.add(_line.replace('\n', ''))
    with open(_file_name) as f:
        for _line in f.readlines():
            if random() < 0.1 and _rand:
                pass
            _line = _line.replace('\n', '').split(',')
            _emotion, _words = _line[0], _line[1].split(' ')
            _words = filter(lambda x: x not in _stop_words, _words)
            yield _emotion, list(_words)


def classification():
    emotions = {
        'anger': Emotion(),
        'disgust': Emotion(),
        'fear': Emotion(),
        'guilt': Emotion(),
        'joy': Emotion(),
        'sadness': Emotion(),
        'shame': Emotion()
    }
    for label, words in read_file('data/train.txt', True):
        emotions[label].appear()
        for word in words:
            emotions[label].add(word)
    a, b = 0, 0
    for label, words_list in read_file('data/test.txt'):
        max_p = 0
        for name, emotion in emotions.items():
            p = emotion.probability
            for word in words_list:
                p *= emotion.get_word_probability(word, laplace_smoothing=True, alpha=0.001)
            if p > max_p:
                max_p = p
                max_p_emotion = name
        if max_p_emotion == label:
            a += 1
        else:
            b += 1
        print(a / (a + b))
        yield max_p_emotion


if __name__ == '__main__':
    with open('084_3.txt', 'wt') as f:
        for emotion in classification():
            f.write(emotion + '\n')
