
stopWords = []

def getStopWordList(stopWordListFileName):
        stopWords = []
        stopWords.append('AT_USER')
        stopWords.append('URL')

        fp = open(stopWordListFileName, 'r')
        line = fp.readline()
        while line:
                word = line.strip()
                stopWords.append(word)
                line = fp.readline()
        fp.close()
        return stopWords

# st = open('stopwords.txt', 'r')
stopWords = getStopWordList('stopwords.txt')

def compStopWords(fv):
        fvr = []
        # print stopWords
        for w in fv:
            if(w in stopWords):
                # print w
                continue
            else:
                fvr.append(w)
        # print fvr
        return fvr
