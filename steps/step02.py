import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func

    def backward(self):
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input,f.output
            x.grad=




class Function:
    #클래스의 객체도 호출하게 해주는 함수
    #f=Function(), f(x)로 호출 가능
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        output.set_creator(self)
        self.input=input
        self.output=output
        return output

    def forward(self,in_data):
        raise NotImplementedError()

    def backward(self,gy):
        raise NotImplementedError()



class Square(Function):
    def forward(self,x):
        y= x**2
        return y

    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx


class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y

    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
