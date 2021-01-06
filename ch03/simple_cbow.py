import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul,SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H=vocab_size,hidden_size      #변수 초기화

        #가중치 초기화
        W_in=0.01*np.random.randn(V,H).astype('f')  #V,H행렬을 float타입으로 랜덤으로 만들어 가중치로 만든다.
        W_out=0.01*np.random.randn(H,V).astype('f')

        #계층 생성
        self.in_layer0=MatMul(W_in)
        self.in_layer1=MatMul(W_in)

