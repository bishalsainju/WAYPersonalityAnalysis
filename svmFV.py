import json
import csv
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

from ps import processStatus
from gfv import getFeatureVector
from sw  import compStopWords
from stem import stemm
from wcntr import cntr

# def featExtr():
with open('analysisFiles/usermap.json') as json_data:
    usermap = json.load(json_data)
with open('analysisFiles/userNeu.json') as json_data1:
    userNeu = json.load(json_data1)
# with open('analysisFiles/userCon.json') as json_data2:
#     userCon = json.load(json_data2)
# with open('analysisFiles/userExt.json') as json_data3:
#     userExt = json.load(json_data3)
# with open('analysisFiles/userAgr.json') as json_data4:
#     userAgr = json.load(json_data4)
# with open('analysisFiles/userNeu.json') as json_data5:
#     userNeu = json.load(json_data5)
#
# f = open("analysisFiles/totalBOW.txt", "r")
# lines = f.read().split('\n')
# lines = lines[1:-1]
# f.close()

# totalbagofwords = set()
# totalbagofwords = list(lines)
# totalbagofwords.sort()

training_feature_set = []
test_set = []
train_labels = []
test_labels = []
training_users = set()

TAKE = 2220
cnt = 0

for id in usermap:
    training_users.add(id)
    cnt += 1
    if cnt <= TAKE:
        training_feature_set.append(' '.join(usermap[id]))
        train_labels.append(userNeu[id])
    else:
        test_set.append(' '.join(usermap[id]))
        test_labels.append(userNeu[id])

pred_set = []
txt = raw_input("Enter a sentence: ")
processedStatus = processStatus(txt)
a1 = getFeatureVector(processedStatus)
final1 = compStopWords(a1)
final2 = stemm(final1)
featureVector = compStopWords(final2)
pred_set.append(' '.join(featureVector))

# print training_feature_set
vectorizer = TfidfVectorizer(min_df=1,
                             max_df = 0.95,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_feature_set)
test_vectors = vectorizer.transform(pred_set)

# Perform classification with SVM, kernel=rbf
print train_vectors.shape
print test_vectors.shape
# print train_vectors
# print test_vectors

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
print prediction_linear
# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

# # Print results in a nice table
# print("Results for SVC(kernel=rbf)")
# print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
# print(classification_report(test_labels, prediction_rbf))
# print("Results for SVC(kernel=linear)")
# print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
# print(classification_report(test_labels, prediction_linear))
# print("Results for LinearSVC()")
# print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
# print(classification_report(test_labels, prediction_liblinear))
