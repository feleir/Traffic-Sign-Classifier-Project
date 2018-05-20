# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[trainingdataset]: ./writeup-images/training-dataset.png "Training dataset"
[validationdataset]: ./writeup-images/validation-dataset.png "Validation dataset"
[testdataset]: ./writeup-images/test-dataset.png "Test dataset"
[speedlimit]: ./writeup-images/speed-limit.png "Speed limit samples"
[priorityroad]: ./writeup-images/priority-road.png "Priority road samples"
[speedlimitprocessed]: ./writeup-images/speed-limit-processed.png "Speed limit processed samples"
[priorityroadprocessed]: ./writeup-images/priority-road-processed.png "Priority road processed samples"
[trainingdatasetuniform]: ./writeup-images/training-dataset-uniform.png "Training dataset augmented"
[validationepoch]: ./writeup-images/validation-epoch.png "Validation increase over epoch"
[downloadedimages]: ./writeup-images/downloaded-images.png "Downloaded images"
[predictions]: ./writeup-images/predictions.png "Softmax Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb), or exported to [html](https://github.com/feleir/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html) and [pdf](https://github.com/feleir/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.pdf)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43, the labels description can be found [here]()https://github.com/feleir/Traffic-Sign-Classifier-Project/blob/master/signnames.csv

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. This bar chart shows the data distribution of the datasets

![Training dataset distribution][trainingdataset]

![Validation dataset distribution][validationdataset]

![Test dataset distribution][testdataset]

As a sample here are some traffic signs from the training data set.

0.speSpeed limit (20km/h) - Samples: 180
![20km/h speed limit signs][speedlimit]
12.Priority road - Samples: 1890
![Priority road sign][priorityroad]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I reduced the images from 3 channels (RGB) to one grayscale only. The main reason why this is needed is because it reduces the amount of input data and makes the model faster, also the color of the traffic sign should not be an important factor for the its classification.

After that I normalized the image data so it has mean zero and equal variance, applied (pixel - 128)/128 as suggested in the project notebook.

As a sample here are some traffic signs after applying grascale and normalization.

0.speSpeed limit (20km/h) - Samples: 180
![20km/h speed limit signs processed][speedlimitprocessed]
12.Priority road - Samples: 1890
![Priority road signs processed][priorityroadprocessed]

As seen in the diagrams before, the distribution of the training dataset is not uniform so I decided to generate random images for labels that had less than **1000** minimum samples, applying random image transformations using the cv2 library.

- Translation
- Scaling
- Warping
- Brightness

This will produce a model not biased for certain solutions.

![Augmented training dataset distribution][trainingdatasetuniform]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, outputs 10x10x16  				|
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Convolution 5x5       | 1x1 stride, outputs 1x1x400                   |
| Flatten               | Flatten 5x5x16 to 400                         |
| Flatten               | Flatten 1x1x400 to 400                        |
| Concatenate           | Both flattens 400 + 400 = 800                 |
| Dropout               | 0.5 while training, 1 on validation           |
| Fully connected		| Output 43    									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model in an EC2 GPU instance using the course tutorial, which really improved the training times from about 30 minutes in my local machine to 5 mins.

This are the final parameters used for training:
- EPOCHS: 50
- BATCH_SIZE: 100
- mu: 0
- sigma 0.1
- learning rate: 0.0009
- keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of **96.1%**
* test set accuracy of **95.8%**

I started using LeNet-5, described in the previous lab, with some modifications to be able to use a 32x32x1 input and an output of 43. Tried several changes in epoch/batch_size/learning rate but couldn't get more that *91%* validation accuracy and around *82%* test accuracy.
After that decided to try use Sermanet/Lecun model based on the paper linked in the notebook, using the idea to add a third convolution layer and combine the flatten results of the second and third convolution layers. This approach provided much better results, specially when lowering the learning rate.

I decided to set an epoch of around *50* as there was not a lot of improvement in the accuracy at that time, as it can be check in the next diagram.

![Validation increase over epoch][validationepoch]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web edited to 32x32 pngs to avoid the need of conversion in the notebook. I choose easy to identify signs to check how the model performs.

![German signs downloaded][downloadedimages]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)   						| 
| Bumpy road    		| Bumpy road 								    |
| Ahead only			| Ahead only									|
| No vehicles	      	| No vehicles					 				|
| Go straight or left	| Go straight or left      						|
| General caution       | General caution


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of *100%*. This compares favorably to the accuracy on the test set of *95.8%*.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As it can be seend in the next image, the model identify provied a probability of 1.0 in all images but the first one, which had a 75% change of 30km/h and a 25% roundabout mandatory. This is one of the labels with less images in the training set so while the augmented data did help with the classification it still didn't give a full probability like the other images.

![Softmax predictions][predictions]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Due to lack of time I wasn't able to complete this.
