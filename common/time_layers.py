# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid

class RNN:
    def __init__(self,Wx,Wh,b):   #가중치 두개와 편향 하나
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache=None #역전파 계산 시 사용하는 중간 데이터 담기

    #아래로부터 입력 x와 왼쪽으로부터의 입력 h_prev를 받는다.
    def forward(self,x,h_prev):
        Wx,Wh,b=self.params
        t=np.matmul(h_prev,Wh)+np.matmul(x,Wx)+b
        h_next=np.tanh(t)

        self.cache=(x,h_prev,h_next)
        return h_next

    def backward(self,dh_next):
        Wx,Wh,b=self.params
        x,h_prev,h_next=self.cache

        dt=dh_next*(1-h_next**2)    #tanh의 역전파
        db=np.sum(dt,axis=0)
        dWh=np.matmul(h_prev.T,dt)
        dh_prev=np.matmul(dt,Wh.T)
        dWx=np.matmul(x.T,dt)
        dx=np.matmul(dt,Wx.T)

        self.grads[0][...]=dWx
        self.grads[1][...]=dWh
        self.grads[2][...]=db

        return dx,dh_prev

class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False):    #은닉 상태를 인계받을지를 stateful이라는 인수로 조정
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers=None

        self.h,self.dh=None,None    #h는 forward()메소드를 불렀을 때 마지막 RNN계층의 은닉 상태 저장
                                    #dh는 backward()를 불렀을 때 하나 앞 블록의 은닉 상태 기울기를 저장
        self.stateful=stateful

    #TimeRNN계층의 은닉상태를 설정
    def set_state(self,h):
        self.h=h

    #은닉 상태를 초기화하는 메소드
    def reset_state(self):
        self.h=None


    def forward(self,xs):
        """
        :param xs: T개 분량의 시계열 데이터를 하나로 모은 것. 미니배치 크기 N, 입력 벡터 차원의 수 D
        xs의 형상 (N,T,D)
        """
        Wx,Wh,b=self.params
        N,T,D=xs.shape  #형상
        D,H=Wx.shape

        self.layers=[]
        hs=np.empty((N,T,H),dtype='f')

        if not self.stateful or self.h is None:
            self.h=np.zeros((N,H),dtype='f')

        for t in range(T):
            layer=RNN(*self.params)
            self.h=layer.forward(xs[:,t,:],self.h)
            hs[:,t,:]=self.h
            self.layers.append(layer)

        return hs


    def backward(self,dhs):
        Wx,Wh,b=self.params
        N,T,H=dhs.shape
        D,H=Wx.shape

        dxs=np.empty((N,T,D),dtype='f')
        dh=0
        grads=[0,0,0]
        for t in reversed(range(T)):
            layer=self.layers[t]
            dx,dh=layer.backward(dhs[:,t,:]+dh)   #합산된 기울기
            dxs[:,t,:]=dx

            for i,grad in enumerate(layer.grads):
                grads[i]+=grad

        for i,grad in enumerate(grads):
            self.grads[i][...]=grad

        self.dh=dh

        return dxs


class TimeEmbedding:
    def __init__(self,W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.layers=None
        self.W=W

    def forward(self,xs):
        N,T=xs.shape
        V,D=self.W.shape

        out=np.empty((N,T,D),dtype='f')
        self.layers=[]

        for t in range(T):
            layer=Embedding(self.W)
            out[:,t,:]=layer.forward(xs[:,t])
            self.layers.append(layer)

        return out

    def backward(self,dout):
        N,T,D=dout.shape

        grad=0
        for t in range(T):
            layer=self.layers[t]
            layer.backward(dout[:,t,:])
            grad+=layer.grads[0]

        self.grads[0][...]=grad
        return None

class TimeAffine:
    def __init__(self,W,b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W),np.zeros_like(b)]
        self.x=None

    def forward(self,x):
        N,T,D=x.shape
        W,b=self.params

        rx=x.reshape(N*T,-1)
        out=np.dot(rx,W)+b
        self.x=x
        return out.reshape(N,T,-1)

    def backward(self,dout):
        x=self.x
        N,T,D=x.shape
        W,b=self.params

        dout=dout.reshape(N*T,-1)
        rx=x.reshape(N*T,-1)

        db=np.sum(dout,axis=0)
        dW=np.dot(rx.T,dout)
        dx=np.dot(dout,W.T)
        dx=dx.reshape(*x.shape)

        self.grads[0][...]=dW
        self.grads[1][...]=db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads=[],[]
        self.cache=None
        self.ignore_label=-1

    def forward(self,xs,ts):
        N,T,V=xs.shape

        if ts.ndim==3:
            ts=ts.argmax(axis=2)

        mask=(ts!=self.ignore_label)

        xs=xs.reshape(N*T,V)
        ts=ts.reshape(N*T)
        mask=mask.reshape(N*T)

        ys=softmax(xs)
        ls=np.log(ys[np.arange(N*T),ts])
        ls*=mask
        loss=-np.sum(ls)
        loss/=mask.sum()

        self.cache=(ts,ys,mask,(N,T,V))
        return loss

    def backward(self,dout=1):
        ts,ys,mask,(N,T,V)=self.cache

        dx=ys
        dx[np.arange(N*T),ts] -=1
        dx*=dout
        dx/=mask.sum()
        dx*=mask[:,np.newaxis]

        dx=dx.reshape((N,T,V))

        return dx

