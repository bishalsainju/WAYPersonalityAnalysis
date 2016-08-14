import enchant
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk import PorterStemmer
import re

Dictionary = enchant.Dict("en_US")

def replaceTwoOrMore(s):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)

def getFeatureVector(status):
        featureVector = []
        postag = nltk.pos_tag(status.split())
        words = [x[0] for x in postag if x[1] not in ["NN", "IN", "CC", "TO", "NNS", "NNP", "NNPS"]]
        for w in words:
                w = replaceTwoOrMore(w)
                # w = w.strip('\'"?,.')
                val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
                if(val is None):
                    continue
                if(Dictionary.check(w) == False):
                        continue
                else:
                    w = w.lower()
                    # w = port.stem(w).encode('ascii')
                    # if(w in stopWords):
                    #         continue
                    # if w in featureVector: featureVector[w] += 1
                    # else: featureVector[w] = 1
                    featureVector.append(w)
        return featureVector
