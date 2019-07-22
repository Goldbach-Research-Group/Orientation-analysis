import re
import csv
import jieba
import numpy as np
from try1.Tools import readFile_JSON		# DataDealer项目下的Tools.py文件
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

link="./data"								# 路径
filename="306537777_20181231_001"			# 答案json文件名
encoding="utf-8"
stpwrdpath="../stopWords/stopWordList(sou).txt"  	# 停用词表路径
analyCSVpath="./data/倾向性分析数据集.csv"  # 已评分数据集(CSV文件)路径

def classi(score):	# 根据评分分成四类
    score=float(score)
    if(score>=0 and score<0.25):
        return 0
    if(score>=0.25 and score<0.5):
        return 1
    if(score>=0.5 and score<0.75):
        return 2
    if(score>=0.75 and score<1):
        return 3

# 读取答案json
ansJson=readFile_JSON(link,filename,encoding)

# 提取答案内容
dr=re.compile(r'<[^>]+>', re.S)  # 提取答案内容用的正则
content=[]
for i in ansJson:
    content.append(dr.sub('',i['content']))
# print(content)

# 答案内容分词
contentCut=[]
for i in content:
    contentCut.append(" ".join(jieba.cut(i)))
# print(contentCut)

# 从文件导入停用词表
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('utf-8')  # 提用词提取
stpwrdlst = stopword.splitlines()   #将停用词表转换为list

# 转换为词向量
vect = CountVectorizer(stop_words=stpwrdlst) # 模型
term_matrix = vect.fit_transform(contentCut) # 得到的词向量

# 读入评分数据集
fp2=open(analyCSVpath,'r',encoding=encoding)
analyCSV=csv.reader(fp2)
X=[] # 答案内容
y=[] # 对应评分
for i in analyCSV:
    X.append(i[1])
    y.append(classi(i[2]))
X=vect.fit_transform(X)

# 训练
X=np.array(X)
y=np.array(y)

clf = GaussianNB()# 默认priors=None，可用clf.set_params设置各个类标记的先验概率
clf.fit(X,y)

fp2.close()