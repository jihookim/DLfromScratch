from common.config import GPU
from common.layers import Embedding
import numpy as np
import collections
from common.layers import Embedding,SigmoidWithLoss

class EmbeddingDot:
    def __init__(self,W):
        self.embed=Embedding(W)
        self.params=self.embed.params
        self.grads=self.embed.grads
        self.cache=None

    def forward(self,h,idx):
        target_W=self.embed.forward(idx)
        out=np.sum(target_W*h,axis=1)

        self.cache=(h,target_W)
        return out

    def backward(self,dout):
        h,target_W=self.cache
        dout=dout.reshape(dout.shape[0],1)

        dtarget_W=dout*h
        self.embed.backward(dtarget_W)
        dh=dout*target_W
        return dh


class UnigramSampler:
    """한 단어를 대상으로 확률분포를 만든다"""
    def __init__(self,corpus,power,sample_size):
        self.sample_size=sample_size
        self.vocab_size=None
        self.word_p=None

        counts=collections.Counter()    #컨테이너에 동일한 값이 몇 개 있는지 파악하는데 사용
        for word_id in corpus:
            counts[word_id]+=1

        vocab_size=len(counts)
        self.vocab_size=vocab_size
        self.word_p=np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i]=counts[i]

        self.word_p=np.power(self.word_p,power)
        self.word_p/=np.sum(self.word_p)

    def get_negative_sample(self,target):
        batch_size=target.shape[0]

        if not GPU:
            negative_sample=np.zeros((batch_size,self.sample_size),dtype=np.int32)

            for i in range(batch_size):
                p=self.word_p.copy()
                target_idx=target[i]
                p[target_idx]=0
                p/=p.sum()
                negative_sample[i,:]=np.random.choice(self.vocab_size,size=self.sample_size,replace=False,p=p)
        else:
            negative_sample=np.random.choice(self.vocab_size,size=(batch_size,self.sample_size),replace=True,p=self.word_p)

        return negative_sample




class NegativeSamplingLoss:
    def __init__(self,W,corpus,power=0.75,sample_size=5):
        self.sample_size=sample_size
        self.sampler=UnigramSampler(corpus,power,sample_size)
        self.loss_layers=[SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers=[EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params,self.grads=[],[]
        for layer in self.embed_dot_layers:
            self.params+=layer.params
            self.grads+=layer.grads


    def forward(self,h,target):
        batch_size=target.shape[0]
        negative_sample=self.sampler.get_negative_sample(target)

