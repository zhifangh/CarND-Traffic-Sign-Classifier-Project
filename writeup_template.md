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


[//]: # "Image References"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zhifangh/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\train_data_dist.png)



![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\valid_data_dist.png)



![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\test_data_dist.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\img_src.png)

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\img_grayscale.png)

As a last step, I normalized the image data because ...

Here is an example of a traffic sign image before and after normalized .

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\img_grayscale.png)

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\output\img_normalized.png)




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU	|  |
| Max pooling	| 2x2 stride,  outputs 5x5x16 |
| Flatten	| output: 5x5x16 = 400 |
| Fully connected		| Input = 400. Output = 120. |
| RELU	|  |
| Dropout	| dropout = 0.5 |
| Fully connected	| Input = 120. Output = 84. |
| RELU	|  |
| Dropout	| dropout = 0.5 |
| Fully connected	| Input = 84. Output = 43. |
| Softmax				| mu = 0, sigma = 0.1 |
|						|												|
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used flowing parameters:

optimizer: AdamOptimizer

batch size: 128

number of epochs: 100

learning rate: 0.001



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy : 0.922
* test set accuracy : 0.900

At the first, I set number of epochs to 10, and the validation set accuracy is about 0.80. then i add the number of epochs, such as 20, 30, 50 and so on. At the end, when I set number of epochs to 100,  the validation set accuracy arrive 0.92.




### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\images\1_Stop.jpg)

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\images\2_turn-left-ahead.jpg)

![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\images\3_80-Km-limit.jpg)



![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\images\4_children_crossing.jpg)



​																						 ![](D:\hzf\udacity\project\CarND-Traffic-Sign-Classifier-Project\images\5_double_curve.png)




#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| turn left ahead | Right-of-way |
| 80-Km-limit	| Road work		|
| children crossing	| Slippery road	|
| double curve	| Ahead only   |

I resize the new images to 32 * 32, and apply same preprocess. However, the accuracy model on internet images is 0.20, only the Stop Sign can be predicted.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.92    | Stop sign   									|
| 0.07 | Turn right ahead |
| 0.01	| No entry	|
| 0.00	     | Road work	|
| 0.00				| Yield |




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


