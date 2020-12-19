def seperate(trainX, trainy, testX, testy):
	LtrainX = []
	Ltrainy = []

	RtrainX = []
	Rtrainy = []

	RtestX = []
	Rtesty = []

	LtestX = []
	Ltesty = []

	for x, y in zip(trainX, trainy):
	    if y.split('_')[-1] == 'L':
	        LtrainX.append(x)
	        Ltrainy.append(y)
	    else:
	        RtrainX.append(x)
	        Rtrainy.append(y)

	for x, y in zip(testX, testy):
	    if y.split('_')[-1] == 'L':
	        LtestX.append(x)
	        Ltesty.append(y)
	    else:
	        RtestX.append(x)
	        Rtesty.append(y)

	return RtrainX, Rtrainy, RtestX, Rtesty, LtrainX, Ltrainy, LtestX, Ltesty