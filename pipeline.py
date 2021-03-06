from sklearn import datasets
iris = datasets.load_iris()

#line is the model
#input = featues petal length etc
x = iris.data
#output = labels orchid type
y = iris.target

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

#create the classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)

#predictions
prediction = my_classifier.predict(x_test)
print(prediction)

#calculate our accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))

#----------------------------------------

#different classifier
from sklearn.neighbors import KNeighborsClassifier
my_new_classifier = KNeighborsClassifier()
my_new_classifier.fit(x_train, y_train)

#predictions
prediction_k = my_new_classifier.predict(x_test)
print(prediction_k)

#calculate our accuracy
print(accuracy_score(y_test, prediction_k))

#----------------------------------------

#custom random classifier
from custom_classifier import  CustomRandomKNN
my_custom_classifier = CustomRandomKNN()
my_custom_classifier.fit(x_train, y_train)

#predictions
prediction_custom_k = my_custom_classifier.predict(x_test)
print(prediction_custom_k)

#calculate our accuracy
print(accuracy_score(y_test, prediction_custom_k))

#----------------------------------------

#custom simple kneighbor algorithm k = 1
from simple_kneighbor import SimpleKNeighbor
my_simple_kneighbor_classifier = SimpleKNeighbor()
my_simple_kneighbor_classifier.fit(x_train, y_train)

#prediction
prediction_simple_kneighbor = my_simple_kneighbor_classifier.predict(x_test)
print(prediction_simple_kneighbor)

#calculate accuracy
print(accuracy_score(y_test, prediction_simple_kneighbor))