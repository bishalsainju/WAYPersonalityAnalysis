import json
import csv
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

with open('/home/bishal/workspace/essayPA/analysisFiles/usermap.json') as json_data:
    usermap = json.load(json_data)
with open('/home/bishal/workspace/essayPA/analysisFiles/userNeu.json') as json_data1:
    userNeu = json.load(json_data1)
with open('analysisFiles/usermap1.json') as json_data2:
    usermap1 = json.load(json_data2)
with open('analysisFiles/userNeu1.json') as json_data3:
    userNeu1 = json.load(json_data3)


training_feature_set = []
test_set = []
train_labels = []
test_labels = []

for id in usermap:
    training_feature_set.append(' '.join(usermap[id]))
    train_labels.append(userNeu[id])

for id1 in usermap1:
    test_set.append(' '.join(usermap1[id1]))
    test_labels.append(userNeu1[id1])

vectorizer = TfidfVectorizer(min_df=1,
                             max_df = 0.95,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_feature_set)
test_vectors = vectorizer.transform(test_set)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_labels)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

clf_NB = MultinomialNB()
t0 = time.time()
clf_NB.fit(train_vectors, train_labels)
t1 = time.time()
prd_NB = clf_NB.predict(test_vectors)
t2 = time.time()
time_NB_train = t1-t0
time_NB_predict = t2-t1

# Print results in a nice table
print("Results for SVC(kernel=rbf)")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_labels, prediction_rbf))
print "Accuracy: " + str(accuracy_score(test_labels, prediction_rbf))
print
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_labels, prediction_linear))
print "Accuracy: " + str(accuracy_score(test_labels, prediction_linear))
print
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))
print "Accuracy: " + str(accuracy_score(test_labels, prediction_liblinear))
print
print("Results for NaiveBayes")
print("Training time: %fs; Prediction time: %fs" % (time_NB_train, time_NB_predict))
print(classification_report(test_labels, prd_NB))
print "Accuracy: " + str(accuracy_score(test_labels, prd_NB))

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["1", "0"], rotation=45)
    plt.yticks(tick_marks, ["1", "0"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
cm = confusion_matrix(test_labels, prediction_liblinear)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
