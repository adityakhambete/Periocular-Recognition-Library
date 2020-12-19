from collections import defaultdict
from numpy import argsort
def getPredictionCat(Lsample, Rsample, modelL, modelR):
    Lp = modelL.predict(Lsample.reshape(1,-1))[0]
    Ltop_values_index = sorted(range(len(Lp)), key=lambda i: Lp[i])[-5:]
    Ltop_values = [Lp[i] for i in argsort(Lp)[-5:]]

    probs = defaultdict(int)
    for clas, prob in zip(Ltop_values_index, Ltop_values):
        probs[clas] += prob


    Rp = modelR.predict(Rsample.reshape(1,-1))[0]
    Rtop_values_index = sorted(range(len(Rp)), key=lambda i: Rp[i])[-5:]
    Rtop_values= [Rp[i] for i in argsort(Rp)[-5:]]

    for clas, prob in zip(Rtop_values_index, Rtop_values):
        probs[clas] += prob

    mx_conf = 0
    plabel = -1
    for clas, prob in probs.items():
        if prob > mx_conf:
            mx_conf = prob
            plabel = clas

    confidence = mx_conf*0.5
    
    return (plabel, confidence)