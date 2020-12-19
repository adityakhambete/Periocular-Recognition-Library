from numpy import save, load, savez_compressed
from tensorflow.keras.models import load_model

from periocular_recognition.addNewLabel import addNewLabel
from periocular_recognition.makeParts import makeParts
from periocular_recognition.encodeLabels import encodeLabels
from periocular_recognition.trainUpdatedModel import trainUpdatedModel
from periocular_recognition.extendDataset import extendDataset
from periocular_recognition.getEmbeddings import getEmbeddings
from periocular_recognition.seperate import seperate
from periocular_recognition.getPersonImages import getPersonImages


def registerNewPerson():
    # Load the old model
    modelR = load_model('Trained models/modelR.h5')
    modelL = load_model('Trained models/modelL.h5')
    # load old dataset
    data = load('temp/ubipr_only_front_single_image_expanded_ten_flattened_embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    trainX, trainy, testX, testy = list(trainX), list(trainy), list(testX), list(testy)
    

    # Add new data into the model
    # getPersonImages('1000', 'temp/new_img/test/', 2)
    # getPersonImages('1000', 'temp/new_img/train/', 5)

    ntestX, ntesty = extendDataset('temp/new_img/test/', 1)
    ntrainX, ntrainy = extendDataset('temp/new_img/train/', 5)
    ntestX, ntrainX = getEmbeddings(ntestX, ntrainX)

    savetrainX, savetrainy, savetestX, savetesty =  trainX + list(ntrainX), trainy + list(ntrainy), testX + list(ntestX), testy + list(ntesty)
    savez_compressed('temp/latest_embeddings.npz', savetrainX, savetrainy, savetestX, savetesty)

    new_data = [ntrainX, ntrainy, ntestX, ntesty]
    modelL, modelR, trainX, trainy, testX, testy = addNewLabel(modelL, modelR, trainX, trainy, testX, testy, new_data)

    RtrainX, Rtrainy, RtestX, Rtesty, LtrainX, Ltrainy, LtestX, Ltesty = seperate(trainX, trainy, testX, testy)

    RtrainX, Rtrainy, RtestX, Rtesty = makeParts(RtrainX, Rtrainy, RtestX, Rtesty)
    LtrainX, Ltrainy, LtestX, Ltesty = makeParts(LtrainX, Ltrainy, LtestX, Ltesty)

    RtrainX, Rtrainy, RtestX, Rtesty, encoderR = encodeLabels(RtrainX, Rtrainy, RtestX, Rtesty)
    LtrainX, Ltrainy, LtestX, Ltesty, encoderL = encodeLabels(LtrainX, Ltrainy, LtestX, Ltesty)

    modelL = trainUpdatedModel(modelL, LtrainX, Ltrainy, LtestX, Ltesty)
    modelR = trainUpdatedModel(modelR, RtrainX, Rtrainy, RtestX, Rtesty)

    save('Latest models/classesL.npy', encoderL.classes_)
    save('Latest models/classesR.npy', encoderR.classes_)
    modelR.save('Latest models/modelR.h5') 
    modelL.save('Latest models/modelL.h5') 