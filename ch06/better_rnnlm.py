import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel

class BetterRnnlm(BaseModel):   #가중치 공유 weight tying
    def __init__(self,vocab_size=10000,wordvec_size=650,hidden_size=650,dropout_ratio=0.5):
        V,D,H=vocab_size,wordvec_size,hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')
        lstm_Wx1=(rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh1=(rn(H,))