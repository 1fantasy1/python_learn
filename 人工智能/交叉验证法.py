import numpy as np
import random

def KFold(X,n_splits,is_shuffle=True,random_state=0):
    random.seed(random_state)
    n_samples = len(X)
    indices = list(range(n_samples))
    if is_shuffle:
        random.shuffle(indices)
    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
    result = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for idx in indices if idx not in test_indices]
        train_X = X[train_indices]
        test_X = X[test_indices]
        result.append((train_X, test_X))
    return result

X = np.array([int(i) for i in input().strip().split()])
n_splits = int(input())
result = KFold(X,n_splits)


for S,T in result:
    print(S,T)


print("debug_begin");
res = []
for _,T in result:
    res += list(T)
if set(res)==set(list(X)) and len(X)==len(res):
    print(True)
else:
    print(False)
print("debug_end");