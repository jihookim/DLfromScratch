import numpy as np

class Variable:
    def __init__(self, data):
        print("Variable init")
        print("data",data)
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
                #data가 ndarray형이 아니면 typeError
        self.data = data
        self.grad = None
        self.creator = None

class Function:
    def __call__(self,*inputs):
        print("Child call")

class Add(Function):

    def forward(self, x0, x1):
        print("Add forward")
        y = x0 + x1
        return y  # Variable 객체 전달, Variable init호출


def add(x0, x1):
    print("add")
    return Add()(x0, x1)

x = Variable(np.array(3.0))
y = add(x, x)