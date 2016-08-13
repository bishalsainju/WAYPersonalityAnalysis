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
with open('analysisFiles/usermap1.json') as json_datax:
    usermap1 = json.load(json_datax)
with open('analysisFiles/userOpn.json') as json_data1:
    userOpn = json.load(json_data1)
with open('analysisFiles/userCon.json') as json_data2:
    userCon = json.load(json_data2)
with open('analysisFiles/userExt.json') as json_data3:
    userExt = json.load(json_data3)
with open('analysisFiles/userAgr.json') as json_data4:
    userAgr = json.load(json_data4)
with open('analysisFiles/userNeu.json') as json_data5:
    userNeu = json.load(json_data5)
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
train_labelsO = []
train_labelsC = []
train_labelsE = []
train_labelsA = []
train_labelsN = []
test_labels = []
training_users = set()

TAKE = 2220
cnt = 0

for id in usermap:
    training_users.add(id)
    cnt += 1
    if cnt <= TAKE:
        training_feature_set.append(' '.join(usermap[id]))
        train_labelsO.append(userOpn[id])
        train_labelsC.append(userCon[id])
        train_labelsE.append(userExt[id])
        train_labelsA.append(userAgr[id])
        train_labelsN.append(userNeu[id])
    else:
        test_set.append(' '.join(usermap[id]))
        # train_labelsO.append(userOpn[id])
        # train_labelsC.append(userCon[id])
        # train_labelsE.append(userExt[id])
        # train_labelsA.append(userAgr[id])
        # train_labelsN.append(userNeu[id])

pred_set = []
txt = raw_input("Write about yourself: ")
processedStatus = processStatus(txt)
a1 = getFeatureVector(processedStatus)
final1 = compStopWords(a1)
final2 = stemm(final1)
featureVector = compStopWords(final2)
# pred_set.append(' '.join(usermap1["5489ed38556af050d6a93e5d27b95dfb"]))
pred_set.append(' '.join(featureVector))
# print training_feature_set
vectorizer = TfidfVectorizer(min_df=1,
                             max_df = 0.95,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_feature_set)
test_vectors = vectorizer.transform(pred_set)

# Perform classification with SVM, kernel=rbf
# print train_vectors.shape
# print test_vectors.shape
# print train_vectors
# print test_vectors

# classifier_rbf = svm.SVC()
# t0 = time.time()
# classifier_rbf.fit(train_vectors, train_labels)
# t1 = time.time()
# prediction_rbf = classifier_rbf.predict(test_vectors)
# t2 = time.time()
# time_rbf_train = t1-t0
# time_rbf_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linearO = svm.SVC(kernel='linear')
classifier_linearC = svm.SVC(kernel='linear')
classifier_linearE = svm.SVC(kernel='linear')
classifier_linearA = svm.SVC(kernel='linear')
classifier_linearN = svm.SVC(kernel='linear')

t0 = time.time()
classifier_linearO.fit(train_vectors, train_labelsO)
classifier_linearC.fit(train_vectors, train_labelsC)
classifier_linearE.fit(train_vectors, train_labelsE)
classifier_linearA.fit(train_vectors, train_labelsA)
classifier_linearN.fit(train_vectors, train_labelsN)
t1 = time.time()
prediction_linearO = classifier_linearO.predict(test_vectors)
prediction_linearC = classifier_linearC.predict(test_vectors)
prediction_linearE = classifier_linearE.predict(test_vectors)
prediction_linearA = classifier_linearA.predict(test_vectors)
prediction_linearN = classifier_linearN.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
print "Opn: " + str(prediction_linearO)
print "Con: " + str(prediction_linearC)
print "Ext: " + str(prediction_linearE)
print "Agr: " + str(prediction_linearA)
print "Neu: " + str(prediction_linearN)
# Perform classification with SVM, kernel=linear
# classifier_liblinear = svm.LinearSVC()
# t0 = time.time()
# classifier_liblinear.fit(train_vectors, train_labels)
# t1 = time.time()
# prediction_liblinear = classifier_liblinear.predict(test_vectors)
# t2 = time.time()
# time_liblinear_train = t1-t0
# time_liblinear_predict = t2-t1

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
