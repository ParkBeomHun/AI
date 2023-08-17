##AI/딥러닝/파이썬공부_박범훈_20230817

import numpy as np

print("============First AND===============")
def AND(x1,x2):
    w1,w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1
    

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))


print("============Second AND===============")
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    elif tmp >0:
        return 1
    
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))



def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    elif tmp >0:
        return 1
    
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.4
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    elif tmp >0:
        return 1
print("============NAND===============")
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))
print("============OR===============")
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))


print("============XOR===============")
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    return AND(s1,s2)

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))