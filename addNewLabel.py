from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Flatten

def updateModel(model):
    # creating a new model
    model_2 = Sequential()

    # getting all the layers except the output one
    for layer in model.layers[:-1]: # just exclude last layer from copying
        model_2.add(layer)

    # prevent the already trained layers from being trained again 
    # (you can use layers[:-n] to only freeze the model layers until the nth layer)
    # for layer in model_2.layers:
    #     layer.trainable = False

    # adding the new output layer, the name parameter is important 
    # otherwise, you will add a Dense_1 named layer, that normally already exists, leading to an error
    num_cats = model.get_layer(index = 1).get_config()['units']
    model_2.add(Dense(num_cats+1, name = 'new_Dense', input_shape=(1024,), kernel_initializer = 'he_uniform', activation = 'softmax'))
    model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_2

def addNewLabel(modelL, modelR, trainX, trainy, testX, testy, new_data):
    #load new dataset
    trainX2, trainy2, testX2, testy2 = new_data

    trainX, trainy, testX, testy = list(trainX), list(trainy), list(testX), list(testy)
    trainX2, trainy2, testX2, testy2 = list(trainX2), list(trainy2), list(testX2), list(testy2)

    trainy += trainy2
    trainX += trainX2
    testy += testy2
    testX += testX2
    
    modelL = updateModel(modelL)
    modelR = updateModel(modelR)
    return modelL, modelR, trainX, trainy, testX, testy