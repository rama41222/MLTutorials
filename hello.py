from sklearn import tree

featues = [[150, 1], [170, 1], [165, 1], [120, 0], [135, 0], [140, 0]]
labels = [2, 2, 2, 3, 3, 3]
clf = tree.DecisionTreeClassifier()
clf.fit(featues, labels)
prediction = clf.predict([[155, 1]])
print(prediction)