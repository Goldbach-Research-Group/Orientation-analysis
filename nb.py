import numpy as np
from sklearn.naive_bayes import GaussianNB # 这里以高斯贝叶斯为例

X = np.array([[-1, -1], [-2, -2], [-3, -3],[-4,-4],[-5,-5], [1, 1], [2,2], [3, 3]])
y = np.array([1, 2, 2,2,3, 1, 0, 0])
clf = GaussianNB()# 默认priors=None，可用clf.set_params设置各个类标记的先验概率
clf.fit(X,y)

print(clf.predict([[-6,-6],[4,5]]))
print(clf.predict_proba([[-6,-6],[4,5]])) # 输出属于各个类的概率
