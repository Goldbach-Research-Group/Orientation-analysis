import csv
from sklearn.feature_extraction.text import CountVectorizer

stpwrdpath=".\\data\\stopWordList.txt"

# 从文件导入停用词表
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('utf-8')  # 提用词提取
#将停用词表转换为list
stpwrdlst = stopword.splitlines()

p=['我 是 个 好人','你 也 是 个 好人','今天 中国 的 天气 非常 好 啊 ！','中国 芯片 技术 落后 于 世界']



vect = CountVectorizer() # 模型
# vect = CountVectorizer(stop_words=stpwrdlst)
term_matrix = vect.fit_transform(p) # 得到的词向量
print(term_matrix.toarray())
print(vect.vocabulary_) # 预览不同词对应的向量维度
