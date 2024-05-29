import numpy as np
A=np.array([[-1,-2.0,-1,1,0],[-2.0,1,-3,0,1]])
b=np.array([[-3],[-4]])
print(b)
Ab = np.append(A, b, axis=1)
b=Ab[:Ab.shape[0],Ab.shape[1]-1]
print(b)