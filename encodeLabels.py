#Normalize input vectors and numerize the labels.
from sklearn.preprocessing import Normalizer, LabelEncoder
def encodeLabels(trainX, trainy, testX, testy):
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(sorted(trainy, key = lambda x: int(x.split('_')[0])))
# 	print(sorted(testy, key = lambda x: int(x.split('_')[0])))
	trainy = out_encoder.transform(sorted(trainy, key = lambda x: int(x.split('_')[0])))
# 	print(trainy)
	testy = out_encoder.transform(sorted(testy, key = lambda x: int(x.split('_')[0])))
# 	print(testy)
	return trainX, trainy, testX, testy, out_encoder
