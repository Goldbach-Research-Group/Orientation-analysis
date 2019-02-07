import numpy as np
from sklearn.naive_bayes import GaussianNB # 这里以高斯贝叶斯为例

X = np.array([[-1, -1,-5], [-2,-6, -2], [-3, -2,-3],[-4,-3,-4],[-5,-2,-5], [1,-1, 1], [2,-3,2], [3,-6, 3]])
y = np.array([1, 2, 2,2,3, 1, 0, 0])
print(X)
clf = GaussianNB()# 默认priors=None，可用clf.set_params设置各个类标记的先验概率
clf.fit(X,y)

# print(clf.predict([[-6,-6],[4,5]]))
# print(clf.predict_proba([[-6,-6],[4,5]])) # 输出属于各个类的概率
