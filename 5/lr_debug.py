import lr

if __name__ == '__main__':
    train_list = list()
    train_list.append(lr.Sample([1, -1], 1))
    train_list.append(lr.Sample([3, 3], 0))
    w = lr.array([1, -2, 3], dtype='float64')
    lr.learning(train_list, w, 1)
    print(w)
    s = lr.Score(w)
    s.test(lr.Sample([-2, 3], 1))
    print(s.accuracy)
