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
        if word in self.__words_count:
            self.__words_count[word] += n
        else:
            self.__words_count[word] = n
        self.__total_words += n
        self.__words_set.add(word)

    @property
    def id(self):
        return self.__emotion_id

    # todo： 这个接口不一定有用
    # @property
    # def size(self):
    #     return self.__total_words

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

    def get_word_probability(self, word, laplace_smoothing=False):
        """
        :param word: 一个测试集单词
        :param laplace_smoothing: 是否开启拉普拉斯平滑
        :return: 返回该单词在该情感中出现的未归一化概率
        """
        if laplace_smoothing is True:
            if word in self.__words_count:
                """
                return self.__words_count[word]
                正确率奇高？？？
                """
                return (self.__words_count[word] + 1) / (self.__total_words + len(self.__words_set))
            else:
                return 1 / (self.__total_words + len(self.__words_set))
        else:
            if word in self.__words_count:
                return self.__words_count[word] / self.__total_words
            else:
                return 0


def classification():
    def read_file(file_name):
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
    rig = 0
    wro = 0
    for answer_id, words_list in read_file('Classification/test.txt'):
        max_probability_id = None
        max_probability = 0
        for emotion_id, emotion in emotions.items():
            p = 1
            for word in words_list:
                p *= emotion.get_word_probability(word, laplace_smoothing=True)
            p *= emotion.probability
            if p > max_probability:
                max_probability = p
                max_probability_id = emotion_id
        if max_probability_id == answer_id:
            print('Hhh', max_probability)
            rig += 1
        else:
            print('ToT', max_probability)
            wro += 1
    print(rig, wro, rig + wro)


class Text(object):
    def __init__(self, emotion_rate, words):
        self.__size = len(words)
        self.__words_count = dict()
        self.__emotion_rate = emotion_rate
        for word in words:
            if word in self.__words_count:
                self.__words_count[word] += 1
            else:
                self.__words_count[word] = 1

    @property
    def words(self):
        return [word for word in self.__words_count]

    def get_word_probability(self, word, laplace_smoothing=False):
        """
        :param word:
        :param laplace_smoothing:
        :return: TF 词频
        """
        if laplace_smoothing is True:
            if word in self.__words_count:
                return (self.__words_count[word] + 1) / (self.__size + len(self.__words_count))
            else:
                return 1 / (self.__size + len(self.__words_count))
        else:
            if word in self.__words_count:
                return self.__words_count[word] / self.__size
            else:
                return 0

    def get_emotion_rate(self, emotion_id):
        return self.__emotion_rate[emotion_id]


def regression():
    def read_file(file_name):
        with open(file_name) as f:
            for line in f.readlines()[1:]:
                line = line.replace('\n', '').split(',')
                yield [float(x) for x in line[2:]], line[1].split(' ')

    train_list = list()
    for emotion_rate, words in read_file('Regression/Dataset_train.csv'):
        train_list.append(Text(emotion_rate, words))
    valid_list = list()
    for emotion_rate, words in read_file('Regression/Dataset_validation.csv'):
        valid_list.append(Text(emotion_rate, words))

    result = list()
    for valid_text in valid_list:
        r_v = list()
        for emotion_id in range(6):
            p = 0
            for test_text in train_list:
                p_t = 1
                for word in valid_text.words:
                    p_t *= test_text.get_word_probability(word, laplace_smoothing=True)
                p += p_t * test_text.get_emotion_rate(emotion_id)
            r_v.append(p)
        total = sum(r_v)
        if total == 0.0:
            r_v = ['0' for rate in r_v]
        else:
            r_v = [str(rate / total) for rate in r_v]
        result.append(r_v)
    with open('reg_res.txt', 'wt') as f:
        for ele in result:
            f.writelines(' '.join(ele) + '\n')


if __name__ == '__main__':
    classification()
    regression()
