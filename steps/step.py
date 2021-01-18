import contextlib

import numpy as np

class Config:
    enable_backprop=True    #역전파 활성화 flag, boolean type

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)


def no_grad():
    return using_config('enable_backprop',False)


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
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        print("Variable backward")
        if self.grad is None:       #grad초깃값은 1
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set=set()

        #funcs에 function추가, seen_set에도 추가
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            print("Variable안에서 f.backward호출")
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    #미분값 같은 변수를 쓴다고 덮어 씌워지지 않기 위해
                    #x.grad가 빈 변수가 아니라면 +하기기
                if x.creator is not None:
                    add_func(x.creator)



def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):    #input은 Variable 클래스
        xs = [x.data for x in inputs]   #input 여러개 받을 수 있게 함
        print("xs는 ",xs)
        ys = self.forward(*xs)          #forward한 결과값 하나
        if not isinstance(ys, tuple):
            ys = (ys,)              #ys튜플로 만들기
        outputs = [Variable(as_array(y)) for y in ys]   #forward한 결과값 outputs에 Variable객체로 다 저장

        if Config.enable_backprop:          #역전파 활성화 상태라면
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)    #output Variable객체마다 함수 알려주기, Function이 아니라 Function의 자식 클래스
            self.inputs = inputs            #Function의 input,output은
            self.outputs = outputs
            print("Function call")
            return outputs if len(outputs) > 1 else outputs[0]  #outputs가 여러개면 outputs리스트,
                                                                #output가 하나면 outputs[0] 변수 전달

    def forward(self, xs):
        print("Function forward")
        raise NotImplementedError()

    def backward(self, gys):
        print("Function backward")
        raise NotImplementedError()



class Add(Function):
    def forward(self, x0, x1):
        print("Add forward")
        y = x0 + x1
        return y #Variable 객체 전달, Variable init호출

    def backward(self, gy):
        print("Add backward")
        return gy, gy



def add(x0, x1):
    print("add")
    return Add()(x0, x1)


x = Variable(np.array(3.0))
y = add(x, x)
print("add 끝 backward시작")
y.backward()
print("x.grad는",x.grad)


x.cleargrad()
y = add(add(x, x), x)
y.backward()
print(x.grad)