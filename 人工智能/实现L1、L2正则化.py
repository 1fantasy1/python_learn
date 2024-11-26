import numpy as np

def L1_Norm(l,theta):
    return l * np.sum(np.abs(theta))

def L2_Norm(l,theta):
    return l * np.sum(np.square(theta))


print("debug_begin");
def test(l,theta):
    print("%.3f"%L1_Norm(l,theta))
    print("%.3f"%L2_Norm(l,theta))
print("debug_end");


l = float(input().strip())
theta = np.array([float(ti) for ti in input().strip().split()])
test(l,theta)