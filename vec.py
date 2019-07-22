from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import stopWords.stop as stop

stop.init('stopWords/stopWordList(gen).txt')

'''
数据预处理：
           分词处理
           切分训练集和测试集
'''
def load_file_and_processing(neg,pos):
    def senToWord(sen): # 分词+去停用词
        sou=list(jieba.cut(sen))
        result = []
        for i in sou:
            if not stop.isStopWord(i):
                result.append(i)
        return result

    pos = [senToWord(i) for i in pos]
    neg = [senToWord(i) for i in neg]

    # fix:这里是二分类，所以积极和消极就是确定的1和0。如果做细粒度还需要改
    y = np.concatenate((np.ones(len(pos)),np.zeros(len(neg)))) # 1是积极，0是消极
    x = np.concatenate((pos,neg))

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    return x_train,x_test,y_train,y_test


'''
对每个句子的所有词向量取均值（用于word2Vec）（fix:这个或许可以改进），生成一个句子的vector
'''
def build_sentence_vector(text,size,imdb_w2v):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1,size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


'''
计算word2Vec词向量
'''
def getWord2Vec(x_train,x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim,min_count=10)    # 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    imdb_w2v.build_vocab(x_train)

    # 在评论集上训练模型
    imdb_w2v.train(sentences=x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.epochs)

    train_vecs = np.concatenate([build_sentence_vector(z,n_dim,imdb_w2v) for z in x_train])
    np.save('train_vecs.npy',train_vecs)
    print('train_vecs size:')
    print(train_vecs.shape)

    # 在测试集上训练
    imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.epochs)
    imdb_w2v.save('w2v_model.pkl')
    # build test tweet vector then scale
    test_vecs = np.concatenate([build_sentence_vector(z,n_dim,imdb_w2v) for z in x_test])
    np.save('test_vecs.npy',test_vecs)
    print('test_vecs size:')
    print(test_vecs.shape)
    return train_vecs,test_vecs,imdb_w2v

stpwrdpath="stopWords/stopWordList(sou).txt"
import os
print(os.path.abspath(stpwrdpath))
# 从文件导入停用词表
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('gbk')  # 提用词提取
stpwrdlst = stopword.splitlines()   #将停用词表转换为list

p=['我 是 个 好人','你 也 是 个 好人','今天 中国 的 天气 非常 好 啊 ！','中国 芯片 技术 落后 于 世界']
vect = CountVectorizer(stop_words=stpwrdlst) # 模型
term_matrix = vect.fit_transform(p) # 得到的词向量
print(term_matrix.toarray())
print(vect.vocabulary_) # 预览不同词对应的向量维度
