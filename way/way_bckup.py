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
pred_set.append(' '.join(usermap1["5489ed38556af050d6a93e5d27b95dfb"]))
pred_set.append(' '.join(featureVector))
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

training_feature_set2 = []
train_labels2 = []
for prediction in prediction_linearO:
    if prediction == 1:
        for id1 in usermap1:
            if float(userOpn2[id1]) >= 3:
                print userOpn2[id1]
                training_feature_set2.append(' '.join(usermap1[id1]))
                train_labels2.append(userOpn2[id1])
    else:
        for id1 in usermap1:
            if float(userOpn2[id1]) <= 3:
                training_feature_set2.append(' '.join(usermap1[id1]))
                train_labels2.append(userOpn2[id1])
vectorizer2 = TfidfVectorizer(min_df=1,
                             max_df = 0.9,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors2 = vectorizer2.fit_transform(training_feature_set2)
test_vectors2 = vectorizer2.transform(pred_set)
classifier_linearO2 = svm.SVC(kernel='linear')
classifier_linearO2.fit(train_vectors2, train_labels2)
prediction_linearO2 = classifier_linearO2.predict(test_vectors2)


print "Classification with SVM, kernel=linear"
print "Opn: " + str(prediction_linearO)
print "Opn: " + str(prediction_linearO2)
# print "Con: " + str(prediction_linearC)
# print "Ext: " + str(prediction_linearE)
# print "Agr: " + str(prediction_linearA)
# print "Neu: " + str(prediction_linearN)
