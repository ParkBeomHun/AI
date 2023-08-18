import numpy as np
import matplotlib.pylab as plt


## step 함수

def step_function(x):
    return np.array(x>0,dtype = int)

x = np.array([-1.0, 1.0,2.0])
print(x)
y = x > 0
y = y.astype(int)
print(y)

print("=====================")


#x= np.arange(-5.0, 5.0, 0.1) # x 값을 -5 ~ 5 0.1간격으로 설정
#y=step_function(x) # y 값을 x가 0보다 클때만 1로 설정
#plt.plot(x,y)
#plt.ylim(-0.1, 1.1) #그래프의 y축 범위를 -0.1 ~ 1.1로 설정
#plt.show()

print("=====================")

## Sigmoid 함수

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array([-1.0,1.0,2.0])
print(sigmoid(x))

t = np.array([1.0,2.0,3.0])
print(1.0 + t)
print(1.0 / t)

x_sigmoid = np.arange(-5.0, 5.0, 0.1)
y_sigmoid=sigmoid(x_sigmoid)            #sigmoid 함수를 위한 x,y 설정
x_step = np.arange(-5.0, 5.0, 0.1)
y_step = step_function(x_step)          #step 함수를 위한 x,y 설정
plt.plot(x_sigmoid,y_sigmoid)
plt.plot(x_step,y_step,linestyle = '--')
plt.ylim(-0.1,1.1)
#plt.show()

print("=====================")

## ReLU 함수
def relu(x):
    return np.maximum(0,x)

x = np.array([-2, -1, 0, 1, 2])
print(relu(x))

print("=====================")

## 다차원 배열 계산

A_1d= np.array([1,2,3,4])
A_2d = np.array([[1,2,3,4],[5,6,7,8]])
print(np.ndim(A_1d))   # n차원 dimension
print(A_1d.shape)      # 해당 배열의 크기를 반환 ex) (2,4)
print(A_1d.shape[0])   # 해당 차원의 갯수를 반환

print(np.ndim(A_2d))   # n차원 dimension
print(A_2d.shape[0])   # shape[0] : 1차원 갯수 반환 -> 2 / shpae[1] : 2차원 갯수 반환 -> 4
                       # 4x3x2x7 배열 = shape[0] : 4 / shape[1] : 3 / shape[2] : 2 / shape[3] : 7

print("=====================")

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(np.dot(A,B))  # 행렬의 곱 
print(A*B)          # 행렬의 같은 위치의 원소끼리의 곱

A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])
print(np.dot(A,B))

print("=====================")

X = np.array([[1,2],[3,4],[5,6]])
Y1 = np.array([1,2])
Y2 = np.array([[1],[2]])
print(np.dot(X,Y1))     # (3x2) x (1x2) = (1x3)
print(np.dot(X,Y2))     # (3x2) x (2x1) = (3x1)

print("=====================")

x = np.array([1,2])
w = np.array([[1,3,5],[2,4,6]])
print(np.dot(x,w))


print("=========3층 신경망 구현 ============")
def identity_function(x):
    return x

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5],[0.2,0.4,0.6]])
b1 = np.array([0.1, 0.2, 0.3])
w2 = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
b2 = np.array([0.1, 0.2])
w3 = np.array([[0.1,0.3],[0.2,0.4]])
b3 = np.array([0.1, 0.2])

a1 = np.dot(x,w1)+b1
z1 = sigmoid(a1)

a2 = np.dot(z1,w2)+b2
z2 = sigmoid(a2)

a3 = np.dot(z2,w3)+b3
z3 = identity_function(a3)

y = z3
print(y)


print("=============소프트맥스함수==============")

a= np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)

y = exp_a/sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a

print(softmax(a))


print("======================================")

a = np.array([1010,1000,990])
print(np.exp(a)) # inf inf inf 로 출력됨 -> 숫자가 너무 커서 안됨
max_a = np.max(a)
sum_exp_a = np.sum(a)
y= (a - max_a)/sum_exp_a
print(y)

a= np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
    
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
