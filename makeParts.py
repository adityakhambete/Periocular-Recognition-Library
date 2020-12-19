#Sort the parts for easy reduction of samples per label.
from numpy import asarray
def makeParts(trainX, trainy, testX, testy):
    zeepTest = sorted(list(zip(testy, testX)), key = lambda x: int(x[0].split('_')[0]))
    zeepTrain = sorted(list(zip(trainy, trainX)), key = lambda x: int(x[0].split('_')[0]))
    trainx = [x for y, x in zeepTrain[:]]
    trainy = [y for y, x in zeepTrain[:]]
    testx = [x for y, x in zeepTest[:]]
    testy = [y for y, x in zeepTest[:]]

    return asarray(trainx), asarray(trainy), asarray(testx), asarray(testy)