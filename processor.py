from ps import processStatus
from gfv import getFeatureVector
from sw  import compStopWords
from stem import stemm
from wcntr import cntr
import csv
import json

inpStatus = csv.reader(open('essays.csv','r'), delimiter = ',', quotechar='"')

usermap = {}
userOpn = {}
userCon = {}
userExt = {}
userAgr = {}
userNeu = {}
totalbagofwords = set()

for row in inpStatus:
    break
for row in inpStatus:
    userid = row[0]
    status = row[1]
    ext = 1 if row[2] == 'y' else 0
    neu = 1 if row[3] == 'y' else 0
    agr = 1 if row[4] == 'y' else 0
    con = 1 if row[5] == 'y' else 0
    opn = 1 if row[6] == 'y' else 0
    processedStatus = processStatus(status)
    a1 = getFeatureVector(processedStatus)
    featureVector = compStopWords(a1)
    # final2 = stemm(final1)
    # featureVector = compStopWords(final2)
    # featureVector = cntr(final3)
    if userid not in usermap:
        usermap[userid] = []
    for word in featureVector:
        totalbagofwords.add(word)
        usermap[userid].append(word)
    userOpn[userid] = round(float(opn))
    userCon[userid] = round(float(con))
    userExt[userid] = round(float(ext))
    userAgr[userid] = round(float(agr))
    userNeu[userid] = round(float(neu))

totalbagofwords = list(totalbagofwords)
totalbagofwords.sort()

writebow = json.dumps(usermap, indent=4)
open("analysisFiles/usermap.json", 'w').write(writebow)

writebowO = json.dumps(userOpn, indent=4)
open("analysisFiles/userOpn.json", 'w').write(writebowO)

# totalbow = {}
# for word in totalbagofwords:
#     totalbow[word] = 0
#
# opStatus = csv.writer(open('wordcountintbow.csv', 'w'), delimiter = ',', quotechar = '"')
# opStatus.writerow(["Word", "Count"])
#
# for word in totalbagofwords:
#     for i in usermap:
#         if word in usermap[i]:
#             totalbow[word] += usermap[i][word]
#     opStatus.writerow([word, int(str(totalbow[word]))])

# for id in usermap:
#     print(id, usermap[id])
#     print "Count: " + str(len(usermap[id]))
#     print("\n")

# print usermap
# totalbagofwords = list(totalbagofwords)
# totalbagofwords.sort()
# for word in totalbagofwords:
#     print word
# writebow = json.dumps(userNeu, indent=4)
# open("userNeu.json", 'w').write(writebow)
