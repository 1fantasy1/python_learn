import numpy as np
from random import seed
from random import randint

'''导入数据'''


def load_moons():
    dataset = []
    line = input()
    while line:
        dx = []
        data = [l for l in line.strip().split(',')]
        dataset.append(np.array([np.float(d) for d in data]))
        line = input()
        if line == "#":
            break
    return dataset


'''划分训练数据与测试数据'''


def split_train_test(dataset, ratio=0.2):
    # ratio = 0.2  # 取百分之二十的数据当做测试数据
    num = len(dataset)
    train_num = int((1 - ratio) * num)
    dataset_copy = list(dataset)
    traindata = list()
    while len(traindata) < train_num:
        index = randint(0, len(dataset_copy) - 1)
        traindata.append(dataset_copy.pop(index))
    testdata = dataset_copy
    return traindata, testdata


def main():
    dataset = load_moons()
    traindata, testdata = split_train_test(dataset)
    X_train = np.array([data[:-1] for data in traindata])
    y_train = np.array([data[-1] for data in traindata])
    X_test = np.array([data[:-1] for data in testdata])
    y_test = np.array([data[-1] for data in testdata])

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    return acc


print("debug_begin")


def test_acc(acc):
    res = True if acc > 0.85 else False
    print(res)


print("debug_end")
acc = main()
test_acc(acc)