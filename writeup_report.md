# **Project 3: Behavioral Cloning** 


---

**Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/CNN.png
[image2]: ./pictures/figure_1.png
[image3]: ./pictures/figure_2.png
[image4]: ./pictures/center.jpg
[image5]: ./pictures/left.jpg
[image6]: ./pictures/right.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/lingyun-wu/CarND-Project-03/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/lingyun-wu/CarND-Project-03/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/lingyun-wu/CarND-Project-03/blob/master/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/lingyun-wu/CarND-Project-03/blob/master/writeup_report.md) summarizing the results
* [video.mp4](https://github.com/lingyun-wu/CarND-Project-03/blob/master/video.mp4) result video 
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia Convolutional Neural Network as my model architecture, which is showed in the image below.
![alt text][image1]

Here is my model summary
```
Number#      Layer (type)               Input Shape            Output Shape          
=======================================================================================
1            Lambda                     (66, 200, 3)           (66, 200, 3)
_______________________________________________________________________________________
2            Conv2D (with Relu)         (66, 200, 3)           (31, 98, 24)
_______________________________________________________________________________________
3            Conv2D (with Relu)         (31, 98, 24)           (14, 47, 36)
_______________________________________________________________________________________
4            Conv2D (with Relu)         (14, 47, 36)           (5, 22, 48)
_______________________________________________________________________________________
5            Conv2D (with Relu)         (5, 22, 48)            (3, 20, 64)
_______________________________________________________________________________________
6            Conv2D (with Relu)         (3, 20, 64)            (1, 18, 64)
_______________________________________________________________________________________
7            Flatten                    (1, 18, 64)            (1164)
_______________________________________________________________________________________
8            Dense (with Relu)          (1164)                 (100)
_______________________________________________________________________________________
9            Dropout (0.5)
_______________________________________________________________________________________
10           Dense (with Relu)          (100)                  (50)
_______________________________________________________________________________________
11           Dropout (0.5)
_______________________________________________________________________________________
12           Dense (with Relu)          (50)                   (10)
_______________________________________________________________________________________
13           Dropout (0.5)
_______________________________________________________________________________________
14           Dense (with Relu)          (10)                   (1)   
```

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting and I only used 5 epochs for the first time training and 3 epochs for refinement training. 

Here are model mean squared error loss change figures for these two trainings

![alt text][image2]
![alt text][image3]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### 4. Creation of the Training Set

The simulator produces three images per frame in the training model. Besides the center image, I also included images from left and right mounted cameras as the data set, which are adjusted by +0.275 for left image and -0.275 for right image. The sample images are showed below

![alt text][image4]  
![alt text][image5]  
![alt text][image6]
 
I also flipped images and angles thinking that this would balance the number of angles of left turn and right turn.


#### 5. Training Process

I split my sample data into 80% training and 20% validation data.

I first recorded two laps of center lane driving and one lane which focused on driving smoothly around curves. Because I'm not very good at driving a car in the game, the car in my recorded video always fluctuates around the center lane. This leads to poor autonomous driving results, in which the car fell off the edge of the first curve.

I then recorded the vehicle recovering from the edge where it fell off in the test run and trained the model again.

Every time the car fell off an edge or stuck somewhere in the track, I recorded the process of vehicle recovering from the edge. This procedure increased the number of the final training data set.

To sum up, I first trained the model with the initial data set, which contains two laps of center lane driving and one lap focusing on curves driving, for 5 epochs. Then, based on this trained model, I collected some data which can help the vehicle recovers from edges. At last, I refined the model with 3 epochs and finally the car could stay on the drivable lane for the whole lap.

The video in the repository is recorded with speed of 20 MPH.  


