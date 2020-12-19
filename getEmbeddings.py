from numpy import asarray, expand_dims
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)
# model.summary()

def extract_embedding(face, model):
    img_data = face.astype('float32')
    img_data = expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)    
    return vgg16_feature


def getEmbeddings(trainX, testX):
    global model
    i = -1
    newTrainX = []
    for face in trainX:
        i += 1 
        if i%100 == 0: 
            print(i/100, end = ' ')
        embedding = extract_embedding(face, model)
        newTrainX.append(embedding.flatten())
    newTrainX = asarray(newTrainX)
    print('')
    print(newTrainX.shape)

    i = -1
    newTestX = []
    for face in testX:
        i += 1
        if i%100 == 0: 
            print(i/100, end = ' ')
        embedding = extract_embedding(face, model)
        newTestX.append(embedding.flatten())
    newTestX = asarray(newTestX)
    print('')
    print(newTestX.shape)
    
    return newTrainX, newTestX