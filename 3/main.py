class Emotion(object):
    # 所有情感在测试集出现的次数，实则是测试集的大小，封装一下方便统计
    __global_count = 0

    def __init__(self, emotion_id):
        self.__emotion_id = emotion_id
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
    def id(self):
        return self.__emotion_id

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


def classification():
    def read_file(file_name):
        """
        分类的分词 generator
        :rtype: emotion_id, [word1, word2, ..., wordn]
        """
        with open(file_name) as f:
            for line in f.readlines()[1:]:
                line = line.replace('\n', '').split(' ')
                yield int(line[1]), line[3:]

    emotions = dict()
    for i in range(1, 7):
        emotions[i] = Emotion(i)
    for emotion_id, words_list in read_file('Classification/train.txt'):
        emotions[emotion_id].appear()
        for word in words_list:
            emotions[emotion_id].add(word)
    right = 0
    wrong = 0
    for answer_id, words_list in read_file('Classification/test.txt'):
        max_probability_id = None
        max_probability = 0
        for emotion_id, emotion in emotions.items():
            p = emotion.probability
            for word in words_list:
                p *= emotion.get_word_probability(word, laplace_smoothing=True, alpha=0.00001)
            if p > max_probability:
                max_probability = p
                max_probability_id = emotion_id
        if max_probability_id == answer_id:
            right += 1
        else:
            wrong += 1
    print(right, wrong, right / (right + wrong))


class Text(object):
    def __init__(self, emotion_rate, words):
        # 不重复词个数
        self.__size = len(words)
        # 每个词出现次数
        self.__words_count = dict()
        # 感情频率
        self.__emotion_rate = emotion_rate
        for word in words:
            if word in self.__words_count:
                self.__words_count[word] += 1
            else:
                self.__words_count[word] = 1

    @property
    def words(self):
        """
        :return: Text 所有词的 list
        """
        return [word for word in self.__words_count]

    def get_word_probability(self, word, laplace_smoothing=False, alpha=1):
        """
        :param alpha: 0-1 优化参数，取决于 label 的个数
        :param word: 单词
        :param laplace_smoothing: 是否开启拉普拉斯平滑
        :return: TF 词频
        """
        if laplace_smoothing is True:
            if word in self.__words_count:
                return (self.__words_count[word] + alpha) / (self.__size + len(self.__words_count) * alpha)
            else:
                return alpha / (self.__size + len(self.__words_count) * alpha)
        else:
            if word in self.__words_count:
                return self.__words_count[word] / self.__size
            else:
                return 0

    def get_emotion_rate(self, emotion_id):
        """
        :param emotion_id: 0-5 六个感情的 id
        :return: 概率
        """
        return self.__emotion_rate[emotion_id]


def regression():
    def read_file(file_name):
        with open(file_name) as f:
            for line in f.readlines()[1:]:
                line = line.replace('\n', '').split(',')
                yield [float(x) for x in line[2:]], line[1].split(' ')

    def normalize(rate_list):
        """
        归一化
        :param rate_list: 情感概率列表
        :return: 归一化后感情概率，为了方便格式化输出，已经转为 string
        """
        rate_sum = sum(rate_list)
        if rate_sum is 0.0:
            return ['0' for rate in rate_list]
        else:
            return [str(rate / rate_sum) for rate in rate_list]

    train_list = list()
    for emotion_rate, words in read_file('Regression/Dataset_train.csv'):
        train_list.append(Text(emotion_rate, words))
    valid_list = list()
    for emotion_rate, words in read_file('Regression/Dataset_validation.csv'):
        valid_list.append(Text(emotion_rate, words))

    with open('reg_res.txt', 'wt') as f:
        for valid_text in valid_list:
            p_list = list()
            for emotion_id in range(6):
                total_p = 0
                for train_text in train_list:
                    tf_p = train_text.get_emotion_rate(emotion_id)
                    for word in valid_text.words:
                        tf_p *= train_text.get_word_probability(word, laplace_smoothing=True, alpha=0.001)
                    total_p += tf_p
                p_list.append(total_p)
            f.writelines(' '.join(normalize(p_list)) + '\n')


if __name__ == '__main__':
    classification()
    regression()
