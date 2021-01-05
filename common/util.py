import numpy as np

def preprocess(text):
    text=text.lower()
    text=text.replace('.',' .')
    words=text.split(' ')

    word_to_id={}
    id_to_word={}
    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word

    corpus=np.array([word_to_id[w] for w in words])

    return corpus,word_to_id,id_to_word



def create_co_matrix(corpus, vocab_size,window_size=1):
    corpus_size=len(corpus)
    co_matrix=np.zeros((vocab_size,vocab_size),dtype=np.int32)

    for idx,word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx=idx-i
            right_idx=idx+i

            if left_idx>=0:
                left_word_id=corpus[left_idx]
                co_matrix[word_id,left_word_id]+=1

            if right_idx<corpus_size:
                right_word_id=corpus[right_idx]
                co_matrix[word_id,right_word_id]+=1

    return co_matrix


def cos_similarity(x,y,eps=1e-8):
    nx=x/np.sqrt(np.sum(x**2)+eps)
    ny=y/np.sqrt(np.sum(y**2)+eps)
    return np.dot(nx,ny)


#어떤 단어가 검색어로 주어지면, 그 검색어와 비슷한 단어를 유사도 순으로 출력하는 함수
#query: 검색어(단어)
#word_matrix: 단어 벡터 행렬
#top: 상위 몇개까지 출력할지 설정
def most_similar(query,word_to_id,id_to_word,word_matrix,top=5):
    # 1.검색어를 꺼낸다.
    if query not in word_to_id:
        print('%s를 찾을 수 없습니다'%query)
        return

    print('\n[query] '+query)
    query_id=word_to_id[query]
    query_vec=word_matrix[query_id]

    #코사인 유사도 계산
    vocab_size=len(id_to_word)
    similarity=np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i]=cos_similarity(word_matrix[i],query_vec)

    #코사인 유사도를 기준으로 내림차순으로 출력
    count=0
    for i in (-1*similarity).argsort():     #(-x).argsort()는 내림차순 출력 메소
        if id_to_word[i]==query:
            continue
        print(' %s: %s' %(id_to_word[i],similarity[i]))

        count+=1
        if count>=top:
            return


#상호정보량: 동시발생 수까지 고려하여 관련성 있는 단어들을 구별하는 척도
#양의 상호정보량: 두 단어의 동시발생 횟수가 0이면 에러나기 때문에 PPMI를 사용
def ppmi(C,verbose=False,eps=1e-8):
    M=np.zeros_like(C,dtype=np.float32)
    N=np.sum(C)
    S=np.sum(C,axis=0)
    total=C.shape[0]*C.shape[1]     #shape[0]: 행 갯수, shape[1]: 열 갯
    cnt=0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[j]*S[i])+eps)
            M[i,j]=max(0,pmi)

            if verbose:
                cnt+=1
                if cnt%(total//100)==0:
                    print('%.1f%% 완료'%(100*cnt/total))

    return M