import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])

#test data
test_data =[0, 50, 100]

#training data
training_target = np.delete(iris.target, test_data)
tranining_data = np.delete(iris.data, test_data, axis=0)

#test data
testing_target = iris.target[test_data]
testing_data = iris.data[test_data]

clf = tree.DecisionTreeClassifier()
clf.fit(tranining_data, training_target)

print(testing_target)
prediction = clf.predict(testing_data)
print(prediction)
