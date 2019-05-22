# Project Writeup

## Traffic Sign classifier

[image1]: ./bar_plot_training.jpg
[image2]: ./bar_plot_validation.jpg
[image3]: ./bar_plot_test.jpg
[image4]: ./probability1.jpg
[image5]: ./probability2.jpg
[image6]: ./probability3.jpg
[image7]: ./probability4.jpg
[image8]: ./probability5.jpg

### Data Set Summary & Exploration


#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I have used **len()** function, **len(np.unique())**, **shape attribute** to get the following summary of the data set.

**Number of training examples = 34799**

**Number of testing examples = 12630**

**Image data shape = (32, 32, 3)**

**Number of classes = 43**

further **collections.Counter()** function has been used to get count for each class in all 3 data sets. This information is further used to plot the bar chart.



#### 2. Include an exploratory visualization of the dataset.

From the analysis of the bar graphs, following things were found:

1. Classes are very imbalanced in all three, train, test and validation, datasets. Some classes are 10 times more than the some other classes.

2. Classes are distributed in similar fashion in all three datasets.

Bar plots of the number of datapoints against each class is included in this jupyter notebook. Also included here.

![training_bar_plot][image1]
![Validation_bar_plot][image2]
![test_bar_plot][image3]


### Design and Test a Model Architecture

I just chose to normalize the datasets. For that I used Min-Max scaling to normalize the data. Normalizing the data brings everything on the same scale which between 0 to 1 in Min-Max scaling which further speeds up training procedure.

I used LeNet5 model which was given in the CNN module of this course and dropout layer was added in each convolution stage.

Following is the final model.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Dropout	      	    | With keep probability as 0.7 during training 	|
| Convolution 5x5x6x16	| 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Dropout	      	    | With keep probability as 0.7 during training 	|
| Fully connected		| Flatten layer with 400 input and 120 output   |
| RELU					|												|
| Fully connected		| Flatten layer with 120 input and 84 output    |
| RELU					|												|
| Fully connected		| Flatten layer with 84 input and 43 output     |
| Softmax				|         									    |



#### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used **AdamOptimizer** with **batch size = 128**, **epochs = 30** and **learning rate = 0.001** to train the model.

I used following methods to train the model and to reach to the final model.

1. First I trained the model directly on the training set which was giving around 0.9 accuracy on validation set but around 0.997 accuracy on training set. 

2. Then i normalized the data by using Min-Max scaling which improved the validation set accuracy till 0.91 but again it was less than the minimum accuracy of 0.935.

3. So that was clear from the training set and validation set accuracies that model was overfitting.

4. To remove ovefitting, we can have some approaches like **dropout**, **Regularization** or **adding more data**.

5. I chose **dropout** method to reduce the overfitting of the model and chose **keep_prob = 0.7** for training and **keep_prob = 1** for calculating the accuracies.

6. **Dropout** method introduces **dropout layer** in each convolution layer after **maxpooling** layer.

7. FInally I was able to get following accuracies on all three datasets.

**Train Accuracy = 0.998**

**Validation Accuracy = 0.949**

**Test Accuracy = 0.938**



### Test a Model on New Images

I downloaded 5 images from the internet. All images werehaving different dimensions as compared to the original dataset image dimension of 32x32x3. So I used **cv2.resize** method to get the required input dimension.

Following 5 images were downloaded.

1. 40_roundabout_mandatory.png 

2. 39_keep_left.png 

3. 35_Ahead_only.png 

4. 38_keep_right.png 

5. 33_turn_right_ahead.png

All 5 images are given in the jupyter notebook.

Further all 5 images were normalized by using Min-Max scaling.

Then accuracy was measured by using saved model and model could easily recognize the all 5images.

downloaded images were very clear and standard images and there was no noise in the images so model could easily able to give 100% accuracy on the new images which is very much inline with accuracy on the test set.

Softmax probablities were calculated by using logits for the new images. Visualization is given in this notebook.

From the graphs it is evident that probability of correct predicted class is much much higher than the other classes. Folloiwing are the graphs of all 5 images.

![image_1][image4]
![image_2][image5]
![image_3][image6]
![image_4][image7]
![image_5][image8]