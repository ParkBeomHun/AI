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
