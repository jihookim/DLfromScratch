import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess,create_contexts_target,convert_one_hot

window_size=1
hidden_size=5
batch_size=3
max_epoch=1000

text='You say goodbye and I say hello.'
corpus,word_to_id,id_to_word=preprocess(text)
#corpus: [0 1 2 3 4 1 5 6]
#word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
#id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}


vocab_size=len(word_to_id) #7
contexts,target=create_contexts_target(corpus,window_size)
"""contexts
[[0 2]
 [1 3]
 [2 4]
 [3 1]
 [4 5]
 [1 6]]"""
#target: [1 2 3 4 1 5]

target=convert_one_hot(target,vocab_size)
contexts=convert_one_hot(contexts,vocab_size)
"""target:
[[0 1 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 0 1 0 0 0]
 [0 0 0 0 1 0 0]
 [0 1 0 0 0 0 0]
 [0 0 0 0 0 1 0]]
 
 contexts:
[[[1 0 0 0 0 0 0]
  [0 0 1 0 0 0 0]]

 [[0 1 0 0 0 0 0]
  [0 0 0 1 0 0 0]]

 [[0 0 1 0 0 0 0]
  [0 0 0 0 1 0 0]]

 [[0 0 0 1 0 0 0]
  [0 1 0 0 0 0 0]]

 [[0 0 0 0 1 0 0]
  [0 0 0 0 0 1 0]]

 [[0 1 0 0 0 0 0]
  [0 0 0 0 0 0 1]]]"""

model=SimpleCBOW(vocab_size,hidden_size)

optimizier=Adam()
trainer=Trainer(model,optimizier)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()


