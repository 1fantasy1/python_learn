import math
import numpy as np
import random
import  warnings

def load_wine():
    X = []
    y = []
    line = input()
    while line:
        dx = []
        data = [l for l in line.strip().split(',')]
        X.append(np.array([np.float(d) for d in data[1:]]))
        y.append(np.int(data[0]))
        line = input()
        if line  == "#":
            break
    return np.array(X),np.array(y)

def train_test_split(X,Y,test_size=0.2,random_state=2333):
    random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    train_indexs = list(set(random.sample(indices.tolist(),int(n_samples*(1-test_size)))))
    test_indexs = [k for k in indices if k not in train_indexs]
    return X[train_indexs],X[test_indexs],Y[train_indexs],Y[test_indexs]
X,y = load_wine()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2333)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("debug_begin");
def test(acc):
    res = True if acc>=0.8 else False
    print(res)
print("debug_end");

test(acc)