def cntr(fa):
    fb = {}
    for w in fa:
        if w in fb: fb[w] += 1
        else: fb[w] = 1
    return fb
