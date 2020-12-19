#Fit for new label classification
from keras.utils.np_utils import to_categorical
import time
from numpy import asarray
def trainUpdatedModel(model, trainX, trainy, testX, testy):
    Y_train = to_categorical(trainy)
    Y_test = to_categorical(testy)
    
    t1 = time.time()
    history = model.fit(asarray(trainX), Y_train,validation_data = (asarray(testX),Y_test), epochs=50, batch_size=64)
    t2 = time.time()

    print("Time taken:", t2-t1)
    return model