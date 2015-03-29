from sklearn import svm
import random

X=[[i] for i in range(100)]
Y=[random.random()>0.5-((i-50)/100) for i in range(100)]


#X = [[1], [2], [3], [10], [4], [5], [20]]
#Y = [False, False, False, False, True, True, True]

#clf = svm.SVC(kernel='linear')
clf = svm.SVR(kernel='linear', degree = 1, verbose = True, max_iter = 100000)
clf.fit(X, Y)
print(clf)
print(clf.predict([[i] for i in range(100) if i%10==0]))
