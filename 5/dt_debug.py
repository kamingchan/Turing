import dt

if __name__ == '__main__':
    ls = [dt.Sample([], 1)] * 9
    ls.extend([dt.Sample([], 0)] * 5)
    d = dt.entropy(ls)
    print(d)
