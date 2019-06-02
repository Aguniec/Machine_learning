import sys, scipy, numpy, matplotlib, pandas, sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = numpy.loadtxt("iris.csv", delimiter = ",")

X = dataset[:,0:4]
Y = dataset[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_valiation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed)
#Test options and evaluation metric
scoring = "accuracy" #ratio od the number of the corectly predicted instances in divided by the total number
models = []
models.append(("LR", LogisticRegression(solver="liblinear",multi_class="ovr"))) #linear regression, linear algorithm
models.append(("LDA", LinearDiscriminantAnalysis())) #Linear Disciminant Analysis, linear algorithm
models.append(("KNN", KNeighborsClassifier())) #K-Nearest Neighbors, nolinear
models.append(("CART", DecisionTreeClassifier())) #Classification and Regression Trees, nolinear
models.append(("NB", GaussianNB())) #Gaussian Naive Bayes
models.append(("SVM", SVC(gamma="auto"))) #Support Vector Machines, nolinear
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
#Compare algorithms
fig = plt.figure()
fig.suptitle("Algorithms Comparsion")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#Make predicions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predicions = knn.predict(X_validation)
print(accuracy_score(Y_valiation, predicions))
print(confusion_matrix(Y_valiation, predicions))
print(classification_report(Y_valiation, predicions))
