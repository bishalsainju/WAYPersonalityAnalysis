from nltk import PorterStemmer

port = PorterStemmer()

def stemm(fa):
    fb = []
    for w in fa:
        w = port.stem(w).encode('ascii')
        fb.append(w)
    return fb
