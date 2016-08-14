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

with open('analysisFiles/usermap.json') as json_data:
    usermap = json.load(json_data)
with open('analysisFiles/usermap1.json') as json_datax:
    usermap1 = json.load(json_datax)
with open('analysisFiles/userOpn.json') as json_data1:
    userOpn = json.load(json_data1)
with open('analysisFiles/userOpn2.json') as json_data10:
    userOpn2 = json.load(json_data10)
with open('analysisFiles/userCon.json') as json_data2:
    userCon = json.load(json_data2)
with open('analysisFiles/userCon2.json') as json_data20:
    userCon2 = json.load(json_data20)
with open('analysisFiles/userExt.json') as json_data3:
    userExt = json.load(json_data3)
with open('analysisFiles/userExt2.json') as json_data30:
    userExt2 = json.load(json_data30)
with open('analysisFiles/userAgr.json') as json_data4:
    userAgr = json.load(json_data4)
with open('analysisFiles/userAgr2.json') as json_data40:
    userAgr2 = json.load(json_data40)
with open('analysisFiles/userNeu.json') as json_data5:
    userNeu = json.load(json_data5)
with open('analysisFiles/userNeu2.json') as json_data50:
    userNeu2 = json.load(json_data50)

training_feature_set = []
train_labelsO = []
train_labelsC = []
train_labelsE = []
train_labelsA = []
train_labelsN = []

for id in usermap:
        training_feature_set.append(' '.join(usermap[id]))
        train_labelsO.append(userOpn[id])
        train_labelsC.append(userCon[id])
        train_labelsE.append(userExt[id])
        train_labelsA.append(userAgr[id])
        train_labelsN.append(userNeu[id])

pred_set = []
# txt = raw_input("Write about yourself: ")
# processedStatus = processStatus(txt)
# a1 = getFeatureVector(processedStatus)
# final1 = compStopWords(a1)
# final2 = stemm(final1)
# featureVector = compStopWords(final2)
pred_set.append(' '.join(usermap1["a6336ec5e11839ae33aee01fa2163652"]))
# pred_set.append(' '.join(featureVector))
vectorizer = TfidfVectorizer(min_df=4,
                             max_df = 0.80,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_feature_set)
test_vectors = vectorizer.transform(pred_set)

# Perform classification with SVM, kernel=linear
classifier_linearO = svm.SVC(kernel='linear')
classifier_linearC = svm.SVC(kernel='linear')
classifier_linearE = svm.SVC(kernel='linear')
classifier_linearA = svm.SVC(kernel='linear')
classifier_linearN = svm.SVC(kernel='linear')
classifier_linearO.fit(train_vectors, train_labelsO)
classifier_linearC.fit(train_vectors, train_labelsC)
classifier_linearE.fit(train_vectors, train_labelsE)
classifier_linearA.fit(train_vectors, train_labelsA)
classifier_linearN.fit(train_vectors, train_labelsN)
prediction_linearO = classifier_linearO.predict(test_vectors)
prediction_linearC = classifier_linearC.predict(test_vectors)
prediction_linearE = classifier_linearE.predict(test_vectors)
prediction_linearA = classifier_linearA.predict(test_vectors)
prediction_linearN = classifier_linearN.predict(test_vectors)

training_feature_setO2 = []
train_labelsO2 = []
for prediction in prediction_linearO:
    if prediction == 1:
        for id1 in usermap1:
            if float(userOpn2[id1]) >= 3:
                training_feature_setO2.append(' '.join(usermap1[id1]))
                train_labelsO2.append(userOpn2[id1])
    else:
        for id1 in usermap1:
            if float(userOpn2[id1]) <= 3:
                training_feature_setO2.append(' '.join(usermap1[id1]))
                train_labelsO2.append(userOpn2[id1])
vectorizerO2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectorsO2 = vectorizerO2.fit_transform(training_feature_setO2)
test_vectorsO2 = vectorizerO2.transform(pred_set)
classifier_linearO2 = svm.SVC(kernel='linear')
classifier_linearO2.fit(train_vectorsO2, train_labelsO2)
prediction_linearO2 = classifier_linearO2.predict(test_vectorsO2)

training_feature_setC2 = []
train_labelsC2 = []
for prediction in prediction_linearC:
    if prediction == 1:
        for id1 in usermap1:
            if float(userCon2[id1]) >= 3:
                training_feature_setC2.append(' '.join(usermap1[id1]))
                train_labelsC2.append(userCon2[id1])
    else:
        for id1 in usermap1:
            if float(userCon2[id1]) <= 3:
                training_feature_setC2.append(' '.join(usermap1[id1]))
                train_labelsC2.append(userCon2[id1])
vectorizerC2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectorsC2 = vectorizerC2.fit_transform(training_feature_setC2)
test_vectorsC2 = vectorizerC2.transform(pred_set)
classifier_linearC2 = svm.SVC(kernel='linear')
classifier_linearC2.fit(train_vectorsC2, train_labelsC2)
prediction_linearC2 = classifier_linearC2.predict(test_vectorsC2)

training_feature_setE2 = []
train_labelsE2 = []
for prediction in prediction_linearE:
    if prediction == 1:
        for id1 in usermap1:
            if float(userExt2[id1]) >= 3:
                training_feature_setE2.append(' '.join(usermap1[id1]))
                train_labelsE2.append(userExt2[id1])
    else:
        for id1 in usermap1:
            if float(userExt2[id1]) <= 3:
                training_feature_setE2.append(' '.join(usermap1[id1]))
                train_labelsE2.append(userExt2[id1])
vectorizerE2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectorsE2 = vectorizerE2.fit_transform(training_feature_setE2)
test_vectorsE2 = vectorizerE2.transform(pred_set)
classifier_linearE2 = svm.SVC(kernel='linear')
classifier_linearE2.fit(train_vectorsE2, train_labelsE2)
prediction_linearE2 = classifier_linearE2.predict(test_vectorsE2)

training_feature_setA2 = []
train_labelsA2 = []
for prediction in prediction_linearA:
    if prediction == 1:
        for id1 in usermap1:
            if float(userAgr2[id1]) >= 3:
                training_feature_setA2.append(' '.join(usermap1[id1]))
                train_labelsA2.append(userAgr2[id1])
    else:
        for id1 in usermap1:
            if float(userAgr2[id1]) <= 3:
                training_feature_setA2.append(' '.join(usermap1[id1]))
                train_labelsA2.append(userAgr2[id1])
vectorizerA2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectorsA2 = vectorizerA2.fit_transform(training_feature_setA2)
test_vectorsA2 = vectorizerA2.transform(pred_set)
classifier_linearA2 = svm.SVC(kernel='linear')
classifier_linearA2.fit(train_vectorsA2, train_labelsA2)
prediction_linearA2 = classifier_linearA2.predict(test_vectorsA2)

training_feature_setN2 = []
train_labelsN2 = []
for prediction in prediction_linearN:
    if prediction == 1:
        for id1 in usermap1:
            if float(userNeu2[id1]) >= 3:
                training_feature_setN2.append(' '.join(usermap1[id1]))
                train_labelsN2.append(userNeu2[id1])
    else:
        for id1 in usermap1:
            if float(userNeu2[id1]) <= 3:
                training_feature_setN2.append(' '.join(usermap1[id1]))
                train_labelsN2.append(userNeu2[id1])
vectorizerN2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectorsN2 = vectorizerN2.fit_transform(training_feature_setN2)
test_vectorsN2 = vectorizerN2.transform(pred_set)
classifier_linearN2 = svm.SVC(kernel='linear')
classifier_linearN2.fit(train_vectorsN2, train_labelsN2)
prediction_linearN2 = classifier_linearN2.predict(test_vectorsN2)

print "Classification with SVM, kernel=linear"
# print "Opn: " + str(prediction_linearO)
print "Opn: " + str(prediction_linearO2)
# print "Con: " + str(prediction_linearC)
print "Con: " + str(prediction_linearC2)
# print "Ext: " + str(prediction_linearE)
print "Ext: " + str(prediction_linearE2)
# print "Agr: " + str(prediction_linearA)
print "Agr: " + str(prediction_linearA2)
# print "Neu: " + str(prediction_linearN)
print "Neu: " + str(prediction_linearN2)
