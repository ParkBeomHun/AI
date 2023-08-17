##AI/딥러닝/파이썬공부_박범훈_20230811


## 자료형
print(type(10))
print(type(3.14))
print(type("Hello"))

##변수
x=10
print(x)
x=100
print(x)
y=3.14
print(x*y)
print(type(x*y))

##리스트 : 배열
a = [1,2,3,4,5]
print(len(a))
print(a[0])
a[4] = 99
print(a)
print(a[0:2]) # 0 ~ 2 미만 범위의 data 출력
print(a[1:])
print(a[:3])
print(a[:-2]) # 마지막에서 2개 빼고 다 출력

##딕셔너리 : 사전같이 data가 이름을 가짐
me = {'height':180} # 처음 dict를 만들때는 {} 안에 각 ''에 해당하는 값을 : 로 지정
me['weight']=70 # 이후에 dict의 항목을 추가할때는 부모노드의 특정부분을 ['']안에 넣고 = 으로 기입
print(me)

##bool, if 문
hungry = True
sleepy = False
print(type(hungry))
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

##for문
for i in[1,3,7]:
    print(i)
      
for i in range(5):
    print(i)

##함수
def hello(object):
    print("Hello", object ,"World!!!")

hello("beautiful")

def plus(num1, num2, num3):
    print("num1 : ",num1," / num2 : ",num2," / num3 : ",num3)
    print(num1+num2+num3)

plus(1,2,3)

##class
class Bum:
    def __init__(self):
        pass
    def setname(self,setname):
        self.name = setname
    def printname(self):
        print("My name is ",self.name,".")


me = Bum()
me.setname("BeomHun")
me.printname()


##넘파이
#그냥 배열 : list
#numpy 배열 : np.array()로 생성하면서 수치계산에 특화되어 더 빠름

import numpy as np      # numpy를 np라는 이름으로 가지고 와라 -> np를 통해 numpy 라이브러리의 메서드참조 가능

x = np.array([1.0 , 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)      ##행과 열의 크기가 같기때문에 연산이됨 *만약 다르다면 에러 발생

print("====================") #3x3x3x2행렬 : 3x3x3 큐브 두개
p = np.array([
    [
    [[1,1,1],[1,1,1],[1,1,1]],
    [[2,2,2],[2,2,2],[2,2,2]],
    [[3,3,3],[3,3,3],[3,3,3]]
    ],
    [
    [[11,11,11],[11,11,11],[11,11,11]],
    [[22,22,22],[22,22,22],[22,22,22]],
    [[33,33,33],[33,33,33],[33,33,33]]
    ]
    ])
print(p)


A = np.array([
    [1,2],[3,4]
    ])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3,0],[0,6]])

print(A+B)
print(A*B)  #같은 위치의 원소끼리의 연산


##브로드캐스트 : 2x2 행렬 x 1x1(스칼라값)이면 1x1의 스칼라값을 2x2로 확장시켜서 같은 위치의 원소끼리 연산

print(A*5)
A1 = [5,10]
A2 = [
    [5],
    [10]
]
print(A*A1)
print(A*A2)

print("================")
A33 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]) #(2x2)*(1x2)는 브로드캐스트를 통해 되지만 (3x3)x(2x1)같은 경우는 브로드 캐스트가 되지않음

for i in range(3):
    print(A33[:,i])

A_flatten = A33.flatten()
print(A_flatten)
print(A_flatten[np.array([0,2,4])])
print(A_flatten>5)
print(A_flatten[A_flatten>5])

#matplotlib
import matplotlib.pyplot as plt

x=np.arange(0,6,0.1)
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle ="--",label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

from matplotlib.image import imread

img = imread('drogba.jpg')
plt.imshow(img)
plt.show()

##바꾸자