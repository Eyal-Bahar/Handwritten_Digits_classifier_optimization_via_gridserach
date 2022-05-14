x1 = [0.93, 0.71]
x0 = [0.35]
import numpy as np

def f(x,w):
    f_x = w*np.log(x) + (1-w)*np.log(1-x)
    return f_x

f_x1 = 0
for x in x1:
    w=1
    f_x1 += f(x,w)

for x in x0:
    w=0
    f_x1 += f(x,w)

print(f_x1)