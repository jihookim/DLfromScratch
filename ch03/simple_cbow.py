import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul,SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H=vocab_size,hidden_size      #변수 초기화

        #가중치 초기화
        W_in=0.01*np.random.randn(V,H).astype('f')
        #V,H행렬을 float타입으로 랜덤으로 만들어 가중치로 만든다.
        W_out=0.01*np.random.randn(H,V).astype('f')

        #계층 생성
        #MatMul class 객체 생성
        self.in_layer0=MatMul(W_in)
        self.in_layer1=MatMul(W_in)
        self.out_layer=MatMul(W_out)
        self.loss_layer=SoftmaxWithLoss()

        #모든 가중치와 기울기를 리스트에 모은다.
        #in_layer:가중치, out_layer:기울기(dx)
        layers=[self.in_layer0,self.in_layer1,self.out_layer]
        self.params, self.grads=[],[]
        for layer in layers:
            self.params+=layer.params
            self.grads+=layer.grads


        #인스턴스 변수에 단어의 분산 표현을 저장
        self.word_vecs=W_in


#맥락(contexts), 타깃(target)을 받아 손실(loss)를 반환
    def forward(self,contexts,target):
        h0=self.in_layer0.forward(contexts[:,0])    #MatMul.forward로 계산
        h1=self.in_layer1.forward(contexts[:,1])
        h=(h0+h1)*0.5
        score=self.out_layer.forward(h)
        loss=self.loss_layer.forward(score,target)
        return loss


#그래프 그대로 backward
    def backward(self,dout=1):
        ds=self.loss_layer.backward(dout)
        da=self.loss_layer.backward(ds)
        da*=0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

