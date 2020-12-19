## Dataset Preparation
 The link to the dataset can be found in the main Readme file. We have worked on the dataset to make it uniform, i.e., with equal dimensions, and ready to be fed into the CNN model that we have used. We have used Transfer Learning in this project to focus more on the other aspects as training images takes up a lot of time. This is talked about later. We further divided the dataset into two proportional datasets that will make the classification easier. 

## Transfer Learning
We have used the VGG-16 model for transfer learning. This is a Convolutional neural network which has been proved to be one of the best multiclassification model and hence was aptly chosen for this project. We have built our own classifier instead of using the existing ones using Artificial Neural Networks. The files have been provided in the repository to build, train, and configure the hyperparameters. 

The prediction process is as follows:

![](https://github.com/adityakhambete/Periocular-Recognition-Library/blob/master/PredictionProcess.PNG)



## Results
The models were trained on 10 samples per participant where the samples were generated using only 2 or 3 existing samples. The accuracy values for different no. of labels can be seen below. 

![](https://github.com/adityakhambete/Periocular-Recognition-Library/blob/master/classification%20accuracy.PNG)

Also the confidence values for the two models are as follows:

![](https://github.com/adityakhambete/Periocular-Recognition-Library/blob/master/Confidence%20Values.PNG)
