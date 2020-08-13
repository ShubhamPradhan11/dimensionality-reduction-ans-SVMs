

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')
data = load_iris()
df = pd.DataFrame(data.data, columns=["sepal length", "sepal width", "petal length", "petal width"])
target_names = ["Iris setosa","Iris virginica" , "Iris versicolor"]
X = data.data
Y = data.target
print(X.shape)

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

x1 = X[0:100]
y1 = Y[0:100]

x1_train, x1_test, y1_train, y1_test = model_selection.train_test_split(x1[:,0:2], y1, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='linear', C=1)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)
plt.scatter(x1[:, 0], x1[:, 1], c=y1, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class setosa and virginica')
plt.show()
print()
target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica")
print(report_lr)

x1_train, x1_test, y1_train, y1_test = model_selection.train_test_split(x1[:,0:2], y1, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='rbf',gamma=.01, C=1)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)
plt.scatter(x1[:, 0], x1[:, 1], c=y1, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class setosa and virginica with kernel = rbf')
plt.show()
print()
target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica with kernel = rbf")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=1000)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)

target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica with kernel = rbf & c = 1000")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=0.001)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)

target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica with kernel = rbf & c =0.001")
print(report_lr)
clf = svm.SVC(kernel='linear', C=0.001)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)
target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica for c=0.001")
print(report_lr)
print()
clf = svm.SVC(kernel='linear', C=1000)
predicted_lr = clf.fit(x1_train,y1_train).predict(x1_test)
target_names_x1 = ["Iris setosa", "Iris virginica"]
con_matrix_lr = metrics.confusion_matrix(y1_test, predicted_lr)
report_lr = metrics.classification_report(y1_test, predicted_lr, target_names = target_names_x1)
print("classification report of setosa and virginica for c = 1000")
print(report_lr)

x2 = X[50:150]
y2 = Y[50:150]

x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(x2[:,0:2], y2, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='linear', C=1)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)
plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class versicolor and virginica')

plt.show()
print()
target_names_x2 = ["Iris virginica" , "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of versicolor and virginica")
print(report_lr)

x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(x2[:,0:2], y2, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='rbf',gamma=.01, C=1)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)
plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class virginica and versicolor with kernel = rbf')
plt.show()
print()
target_names_x2 = ["Iris virginica", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of virginica and versicolor with kernel = rbf")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=1000)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)


con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of virginica and versicolor with kernel = rbf & c = 1000")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=0.001)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)


con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of virginica and versicolor with kernel = rbf & c =0.001")
print(report_lr)

clf = svm.SVC(kernel='linear', C=0.001)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)
target_names_x2 = ["Iris virginica" , "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of versicolor and virginica for c =0.001")
print(report_lr)
print()
clf = svm.SVC(kernel='linear', C=1000)
predicted_lr = clf.fit(x2_train,y2_train).predict(x2_test)
target_names_x2 = ["Iris virginica" , "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y2_test, predicted_lr)
report_lr = metrics.classification_report(y2_test, predicted_lr, target_names = target_names_x2)
print("classification report of versicolor and virginica for c= 1000")
print(report_lr)

a = X[:50,:]
b = X[100:, :]
x3 = np.concatenate((a,b))
a = Y[:50]
b = Y[100:]
y3 = np.concatenate((a,b))

x3_train, x3_test, y3_train, y3_test = model_selection.train_test_split(x3[:,0:2], y3, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='linear', C=1)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)
plt.scatter(x3[:, 0], x3[:, 1], c=y3, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class setosa and versicolor')
plt.show()
print()
target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor")
print(report_lr)





clf = svm.SVC(kernel='linear', C=0.001)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)
target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor for c =0.001")
print(report_lr)
print()
clf = svm.SVC(kernel='linear', C=1000)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)
target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor for c =1000")
print(report_lr)

x3_train, x3_test, y3_train, y3_test = model_selection.train_test_split(x3[:,0:2], y3, test_size=0.3, train_size=0.7)
clf = svm.SVC(kernel='rbf',gamma=.01, C=1)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)
plt.scatter(x3[:, 0], x3[:, 1], c=y3, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('max-margin hyperplane of class setosa and versicolor with kernel = rbf')
plt.show()
print()
target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor with kernel = rbf")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=1000)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)

target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor with kernel = rbf & c = 1000")
print(report_lr)

clf = svm.SVC(kernel='rbf',gamma=.01, C=0.001)
predicted_lr = clf.fit(x3_train,y3_train).predict(x3_test)

target_names_x3 = ["Iris setosa", "Iris versicolor"]
con_matrix_lr = metrics.confusion_matrix(y3_test, predicted_lr)
report_lr = metrics.classification_report(y3_test, predicted_lr, target_names = target_names_x3)
print("classification report of setosa and versicolor with kernel = rbf & c =0.001")
print(report_lr)



