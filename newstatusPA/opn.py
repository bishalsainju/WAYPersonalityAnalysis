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
with open('/home/bishal/workspace/essayPA/analysisFiles/userOpn.json') as json_data1:
    userOpn = json.load(json_data1)
with open('analysisFiles/usermap1.json') as json_data2:
    usermap1 = json.load(json_data2)
with open('analysisFiles/userOpn1.json') as json_data3:
    userOpn1 = json.load(json_data3)
with open('analysisFiles/userOpn2.json') as json_data4:
    userOpn2 = json.load(json_data4)


training_feature_set = []
test_set = []
train_labels = []
test_labels = []

for id in usermap:
    training_feature_set.append(' '.join(usermap[id]))
    train_labels.append(userOpn[id])

for id1 in usermap1:
    test_set.append(' '.join(usermap1[id1]))
    test_labels.append(userOpn1[id1])

vectorizer = TfidfVectorizer(min_df=1,
                             max_df = 0.95,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(training_feature_set)
test_vectors = vectorizer.transform(test_set)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, train_labels)
prediction_linear = classifier_linear.predict(test_vectors)

training_feature_set2 = []
train_labels2 = []
test_labels2 = []
TAKE = 200
cnt = 0
for prediction in prediction_linear:
    if prediction_linear == 1:
        for id in usermap1:
            cnt += 1
            if cnt <= TAKE:
                training_feature_set2.append(' '.join(usermap[id]))
                train_labels.append(userOpn[id])
            else:
                test_set.append(' '.join(usermap[id]))
                test_labels.append(userOpn[id])



# Print results in a nice table
print("Results for SVC(kernel=linear)")
print(classification_report(test_labels, prediction_linear))
print "Accuracy: " + str(accuracy_score(test_labels, prediction_linear))
print

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
cm = confusion_matrix(test_labels, prediction_linear)
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
