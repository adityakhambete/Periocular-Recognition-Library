from numpy import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

from periocular_recognition.getPredictionCat import getPredictionCat
from periocular_recognition.getPersonImages import getPersonImages
from periocular_recognition.getEmbeddings import getEmbeddings
from periocular_recognition.extendDataset import extendDataset


def recognisePerson():
    modelR = load_model('Trained models/modelR.h5')
    modelL = load_model('Trained models/modelL.h5')

    # dummy ID
    # getPersonImages('2345678', 'temp/new_img/current/', 1)


    # Only do the preprocess
    ntestX, ntesty = extendDataset('temp/new_img/current/', 0)
    ntestX, _ = getEmbeddings(ntestX, [])

    encoderL = LabelEncoder()
    encoderL.classes_ = load('Latest models/classesL.npy')
    encoderR = LabelEncoder()
    encoderR.classes_ = load('Latest models/classesR.npy')
    
    label, prob = getPredictionCat(ntestX[0], ntestX[1], modelL, modelR) 
    if prob >= 0.75:
        l, r, p = [encoderL.inverse_transform([label]), encoderR.inverse_transform([label]), prob]
        if l[0][:-2] != r[0][:-2]:
            return("Model Corrupted!")
        else:
            return(l[0][:-2]+ " Confidence: " + str(p))
    else:
        return("Not recognised:", prob)