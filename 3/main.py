class Emotion(object):
    # 所有情感在测试集出现的次数，实则是测试集的大小，封装一下方便统计
    __global_count = 0

    def __init__(self, emotion_id):
        self.__emotion_id = emotion_id
        # 统计每个单词在该情感中出现次数
        self.__words_count = dict()
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

    def get_word_probability(self, word):
        """
        :return: 返回该单词在该情感中出现的未归一化概率
        """
        if word in self.__words_count:
            return self.__words_count[word] / self.__total_words
        else:
            return 0


def read_file(file_name):
    with open(file_name) as f:
        for line in f.readlines()[1:]:
            line = line.replace('\n', '').split(' ')
            yield int(line[1]), line[3:]


def main():
    emotions = dict()
    for i in range(1, 7):
        emotions[i] = Emotion(i)
    for emotion_id, words_list in read_file('Classification/train.txt'):
        emotions[emotion_id].appear()
        for word in words_list:
            emotions[emotion_id].add(word)
    for answer_id, words_list in read_file('Classification/test.txt'):
        max_probability_id = None
        max_probability = 0
        for emotion_id, emotion in emotions.items():
            p = 1
            for word in words_list:
                p *= emotion.get_word_probability(word)
            if p > max_probability:
                max_probability = p
                max_probability_id = emotion_id
        if max_probability_id == answer_id:
            print('Hhh', max_probability)
        else:
            print('ToT', max_probability)


if __name__ == '__main__':
    main()
