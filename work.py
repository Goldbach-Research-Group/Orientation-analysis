import numpy as np
from sklearn.naive_bayes import GaussianNB # 这里以高斯贝叶斯为例
from sklearn.externals import joblib    #把数据转化为二进制
from sklearn.svm import SVC
import vec
import jieba


'''
训练SVM模型
'''
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model.pkl')
    print(clf.score(test_vecs,y_test))
    return clf


'''
构建待测句子的W2v向量
'''
def buildPredictW2v(sen, model):
    allWords = jieba.cut(sen)  # jieba.lcut直接返回list
    train_vecs = vec.buildSentenceW2v(allWords, vec.n_dim, model)
    return train_vecs

'''
构建待测句子的count向量
'''
def buildPredictCountVec(sen,model):
    result = []  # 创建结果向量，model限定维数
    for _ in range(len(model)):
        result.append(0)

    allWords = jieba.cut(sen)  # jieba.lcut直接返回list
    keyList = list(model.keys())
    for word in allWords:
        if word in keyList:
            sub=model[word]
            result[sub]+=1

    return np.array(result)


'''
对单个句子进行情感分析（两个模型都能用）
'''
def predict(words_vecs,clf):
    result = clf.predict(words_vecs)
    probability = clf.predict_proba(words_vecs) # 属于各个类的概率

    if int(result[0]) > 0.5:
        print('positive')
        return True,probability
    else:
        print('negative')
        return False,probability


'''
训练贝叶斯模型
'''
def bayes_train(train_vecs,y_train,test_vecs,y_test):
    clf = GaussianNB()  # 默认priors=None，可用clf.set_params设置各个类标记的先验概率
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model.pkl')
    print(clf.score(test_vecs, y_test))
    return clf


if __name__=='__main__':
    x_train, x_test, y_train, y_test = vec.load_file_and_processing(消极列表,积极列表)
    train_vecs, test_vecs, model = vec.getWord2Vec(x_train, x_test)
    clf = svm_train(train_vecs, y_train, test_vecs, y_test)
    words_vecs=buildPredictW2v('我要好好学习',model)
    predict(words_vecs,clf)
