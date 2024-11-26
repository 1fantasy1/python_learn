def get_acc(y,y_hat):
    acc = sum(1 for yi, yi_hat in zip(y, y_hat) if yi == yi_hat) / len(y)
    return round(acc, 3)

def get_error(y,y_hat):
    return sum(yi != yi_hat for yi,yi_hat in zip(y,y_hat))/len(y)

def get_precision(y,y_hat):
    TP = sum(1 for yi, yi_hat in zip(y, y_hat) if yi == 1 and yi_hat == 1)
    FP = sum(1 for yi, yi_hat in zip(y, y_hat) if yi == 0 and yi_hat == 1)
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

def get_recall(y,y_hat):
    TP = sum(1 for yi, yi_hat in zip(y, y_hat) if yi == 1 and yi_hat == 1)
    FN = sum(1 for yi, yi_hat in zip(y, y_hat) if yi == 1 and yi_hat == 0)
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def get_f1(y,y_hat):
    precision = get_precision(y, y_hat)
    recall = get_recall(y, y_hat)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def get_roc(y,y_hat):
    thresholds = sorted(set(y_hat), reverse=True)
    roc_points = []
    for thresh in thresholds:
        y_pred = [1 if yhat >= thresh else 0 for yhat in y_hat]
        TP = sum(1 for yi, ypi in zip(y, y_pred) if yi == 1 and ypi == 1)
        FP = sum(1 for yi, ypi in zip(y, y_pred) if yi == 0 and ypi == 1)
        TN = sum(1 for yi, ypi in zip(y, y_pred) if yi == 0 and ypi == 0)
        FN = sum(1 for yi, ypi in zip(y, y_pred) if yi == 1 and ypi == 0)
        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0.0
        roc_points.append((FPR, TPR))
    roc_points.append((0.0, 0.0))
    roc_points.append((1.0, 1.0))
    roc_points = sorted(roc_points)
    return roc_points

def get_auc(y,y_hat):
    roc_points = get_roc(y, y_hat)
    roc_points = sorted(roc_points)
    auc = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i - 1]
        x2, y2 = roc_points[i]
        auc += (x2 - x1) * (y1 + y2) / 2  # 梯形面积
    return round(auc, 3)


print("debug_begin");
def test(y,y_hat):
    print("%.3f" %get_acc(y,y_hat))
    print("%.3f" %get_precision(y,y_hat))
    print("%.3f" %get_recall(y,y_hat))
    print("%.3f" %get_f1(y,y_hat))
    print("%.3f" %get_auc(y,y_hat))
print("debug_end");

y = [int(yt) for yt in input().strip().split()]
y_hat = [int(yht) for yht in input().strip().split()]

test(y,y_hat)