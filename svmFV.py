import json
import csv
import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

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
classifier_rbfO = svm.SVC()
classifier_rbfC = svm.SVC()
classifier_rbfE = svm.SVC()
classifier_rbfA = svm.SVC()
classifier_rbfN = svm.SVC()
t0 = time.time()
classifier_rbfO.fit(train_vectors, train_labelsO)
classifier_rbfC.fit(train_vectors, train_labelsC)
classifier_rbfE.fit(train_vectors, train_labelsE)
classifier_rbfA.fit(train_vectors, train_labelsA)
classifier_rbfN.fit(train_vectors, train_labelsN)
t1 = time.time()
prediction_rbfO = classifier_rbfO.predict(test_vectors)
prediction_rbfC = classifier_rbfC.predict(test_vectors)
prediction_rbfE = classifier_rbfE.predict(test_vectors)
prediction_rbfA = classifier_rbfA.predict(test_vectors)
prediction_rbfN = classifier_rbfN.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
print "RBF SVM"
print "Opn: " + str(prediction_rbfO)
print "Con: " + str(prediction_rbfC)
print "Ext: " + str(prediction_rbfE)
print "Agr: " + str(prediction_rbfA)
print "Neu: " + str(prediction_rbfN)

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
print "Linear SVM"
print "Opn: " + str(prediction_linearO)
print "Con: " + str(prediction_linearC)
print "Ext: " + str(prediction_linearE)
print "Agr: " + str(prediction_linearA)
print "Neu: " + str(prediction_linearN)

# Perform classification with SVM, kernel=linear
classifier_liblinearO = svm.LinearSVC()
classifier_liblinearC = svm.LinearSVC()
classifier_liblinearE = svm.LinearSVC()
classifier_liblinearA = svm.LinearSVC()
classifier_liblinearN = svm.LinearSVC()
t0 = time.time()
classifier_liblinearO.fit(train_vectors, train_labelsO)
classifier_liblinearC.fit(train_vectors, train_labelsC)
classifier_liblinearE.fit(train_vectors, train_labelsE)
classifier_liblinearA.fit(train_vectors, train_labelsA)
classifier_liblinearN.fit(train_vectors, train_labelsN)
t1 = time.time()
prediction_liblinearO = classifier_linearO.predict(test_vectors)
prediction_liblinearC = classifier_linearC.predict(test_vectors)
prediction_liblinearE = classifier_linearE.predict(test_vectors)
prediction_liblinearA = classifier_linearA.predict(test_vectors)
prediction_liblinearN = classifier_linearN.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
print "Library Linear SVC"
print "Opn: " + str(prediction_liblinearO)
print "Con: " + str(prediction_liblinearC)
print "Agr: " + str(prediction_liblinearA)
print "Ext: " + str(prediction_liblinearE)
print "Neu: " + str(prediction_liblinearN)

clf_NB_O = MultinomialNB()
clf_NB_C = MultinomialNB()
clf_NB_E = MultinomialNB()
clf_NB_A = MultinomialNB()
clf_NB_N = MultinomialNB()
clf_NB_O.fit(train_vectors, train_labelsO)
clf_NB_C.fit(train_vectors, train_labelsC)
clf_NB_E.fit(train_vectors, train_labelsE)
clf_NB_A.fit(train_vectors, train_labelsA)
clf_NB_N.fit(train_vectors, train_labelsN)
prd_NB_O = clf_NB_O.predict(test_vectors)
prd_NB_C = clf_NB_C.predict(test_vectors)
prd_NB_E = clf_NB_E.predict(test_vectors)
prd_NB_A = clf_NB_A.predict(test_vectors)
prd_NB_N = clf_NB_N.predict(test_vectors)
print "Naive Bayes"
print "Opn: " + str(prd_NB_O)
print "Con: " + str(prd_NB_C)
print "Agr: " + str(prd_NB_E)
print "Ext: " + str(prd_NB_A)
print "Neu: " + str(prd_NB_N)


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
