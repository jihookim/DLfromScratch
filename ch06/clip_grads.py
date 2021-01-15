
#gradients clipping 구현, 기울기 폭발, 기울기 소실 해결

import numpy as np

dW1=np.random.rand(3,3)*10
dW2=np.random.rand(3,3)*10
grads=[dW1,dW2]
max_norm=5.0        #threshold

def clip_grads(grads,max_norm):
    total_norm=0
    for grad in grads:
        total_norm+=np.sum(grad*2)
        #모든 grad의 grad제곱해서 더하기
    total_norm=np.sqrt(total_norm)

    rate=max_norm/(total_norm+1e-6)
    if rate<1:
        for grad in grads:
            grad*=rate
