from math import log


def make_words_list(article_list):
    """
    Make a list of all words in articles.
    :rtype: list
    :param article_list: a list of articles which is a slice by word
    :return: a list of words in the articles (unique)
    """
    words_list = list()
    words_set = set()
    for article in article_list:
        for word in article:
            if word not in words_set:
                words_list.append(word)
                words_set.add(word)
    return words_list


def one_hot(articles_list, words_list):
    def get_vector(text):
        """
        Return a one-hot vector of a text
        :rtype: list
        :param text: a list of word in the text
        :return: one-hot vector
        """
        vector = list()
        for word in words_list:
            if word in text:
                vector.append('1')
            else:
                vector.append('0')
        return vector

    write_str = str()
    one_hot_matrix = list()
    for article in articles_list:
        article_vector = get_vector(article)
        one_hot_matrix.append(article_vector)
        write_str += ' '.join(article_vector) + '\n'
    with open('onehot.txt', 'wt') as file:
        file.write(write_str)
    return one_hot_matrix


def term_frequency(articles_list, words_list):
    write_str = str()
    tf_matrix = list()
    for article in articles_list:
        vector = list()
        for word in words_list:
            vector.append(article.count(word))
        vector = [x / len(article) for x in vector]
        tf_matrix.append(vector)
        vector_str = [str(x) for x in vector]
        write_str += ' '.join(vector_str) + '\n'
    with open('TF.txt', 'wt') as file:
        file.write(write_str)
    return tf_matrix


def tf_idf_product(articles_list, words_list, tf_matrix):
    idf_vector = list()
    for word in words_list:
        total = 1
        for article in articles_list:
            if word in article:
                total += 1
        idf = log(len(articles_list) / total)
        idf_vector.append(idf)
    # with open('IDF.txt', 'wt') as file:
    #     file.write(str(idf_vector))
    write_str = str()
    for row in tf_matrix:
        tfidf_vector = [str(x * y) for x, y in zip(row, idf_vector)]
        write_str += ' '.join(tfidf_vector) + '\n'
    with open('TFIDF.txt', 'wt') as file:
        file.write(write_str)


class SparseMatrix(object):
    def __init__(self, matrix=None):
        self.ele_dict = dict()
        self.ele_set = set()
        if matrix is None:
            self.md = 0
            self.nd = 0
            self.td = 0
        else:
            self.md = len(matrix)
            self.nd = len(matrix[0])
            for i, row in enumerate(matrix):
                for j, element in enumerate(row):
                    if element is not '0':
                        self.ele_set.add((i, j))
                        self.ele_dict[(i, j)] = int(element)
            self.td = len(self.ele_set)


def a_plus_b(one_hot_matrix):
    def save_as_file(matrix, file_name):
        """
        Save a sparese matrix to a text file
        :param matrix: a SparseMatrix
        :param file_name: target file name
        """
        with open(file_name, 'wt') as file:
            file.write('%d\n%d\n%d\n' % (matrix.md, matrix.nd, matrix.td))
            for key, value in sorted(matrix.ele_dict.items()):
                file.write('%d %d %d\n' % (key[0], key[1], value))

    save_as_file(SparseMatrix(one_hot_matrix), 'smatrix1.txt')
    a = SparseMatrix(one_hot_matrix[:623])
    b = SparseMatrix(one_hot_matrix[623:1246])
    assert a.md == b.md and a.nd == b.nd
    c = SparseMatrix()
    c.md = a.md
    c.nd = a.nd
    c.ele_set = a.ele_set | b.ele_set
    for ele in c.ele_set:
        total = 0
        if ele in a.ele_set:
            total += a.ele_dict[ele]
        if ele in b.ele_set:
            total += b.ele_dict[ele]
        c.ele_dict[ele] = total
    c.td = len(c.ele_set)
    save_as_file(c, 'AplusB.txt')


if __name__ == '__main__':
    raw_data = list()
    with open('semeval', 'r') as data:
        for line in data.readlines():
            raw_data.append(line.split('\t')[2].replace('\n', '').split(' '))
    words_list = make_words_list(raw_data)
    one_hot_matrix = one_hot(raw_data, words_list)
    tf_matrix = term_frequency(raw_data, words_list)
    tf_idf_product(raw_data, words_list, tf_matrix)
    a_plus_b(one_hot_matrix)
