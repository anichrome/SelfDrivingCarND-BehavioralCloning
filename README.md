# Writeup Behavioral Cloning Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
Anil Kumar


## Project Goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image_nvidia_model]: ./report/model_architecture.png "Model Visualization"
[center_image]: ./report/center_camera.jpg "Center Training Image"
[left_image]: ./report/left_camera.jpg "Left Training Image"
[right_image]: ./report/right_camera.jpg "Right Training Image"
[center_image_cropped]: ./report/center_camera_cropped.jpg "Cropped Center Training Image"
[center_image_flipped]: ./report/center_camera_flipped.jpg "Flipped Center Training Image"
[left_image_flipped]: ./report/left_camera_flipped.jpg "Flipped Left Training Image"
[right_image_flipped]: ./report/right_camera_flipped.jpg "Flipped Right Training Image"
[video_autonomous_driving]: ./report/video2.gif "Autonomous Driving"
[loss_image]: ./report/loss.png "Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Input Images.

The input images come from three cameras "mounted" on the left, center and right of the car.

![Left Camera Image][left_image] ![Center Camera Image][center_image] ![Right Camera Image][right_image]

As the road should lead driving direction, the images will be cropped to focus on the road parts and then the images are normalized at this stage.

The image below shows the cropped center image.

![Cropped Camera Image][center_image_cropped]


#### Correction of steering angle

The recorded steering angle of left and right camera images must be corrected. It looks like the car is on the right side of the road, if you look at a camera image from the left side, even if the car is in the center of the road.

I have found the best correction value by experiment. I chose the correction value using brute force method and selected the one with the lowest validation loss.

**The correction value I have found is +/- 0.5**


#### Recording

I drove the simulated car several times on track one in forward direction and in reverse direction. Additionally, I recorded only curves and several "rescue" situations to bring the car back to the track. I had to make several runs to get the best data. In the first run, I observed that my training data lacked the "rescue" images and drove constantly to the sides. I then collected more data with these cases and re-run my model.

**I collected in total 26382 training images**, consisting of left, center and right camera images.


#### Image augmentation

I assumed that flipping images would be a useful way to increase the amount of training examples. I have tested this with a simple model. The validation loss decreased a bit, especially in the first few epochs. Further, it helped to have an even distribution of left and right turns. This further increased the amount of training images. Here are the examples of flipped images.

![Left Camera Image Flipped][left_image_flipped] ![Center Camera Image Flipped][center_image_flipped] ![Right Camera Image Flipped][right_image_flipped]

#### My Model

I used the model from NVIDIA and adapted it to reduce overfitting in this work. Before using the one from NVIDIA, I tried with a light weight model consisting of only a few layers. But, instantly I observed that such a model does not perform well the data I generated. Since, the amount of data which I could generate was too small, the small network's performance was poor.

The results from [Convolutional Neuronal Network (CNN) Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from NVIDIA looked promising. I used the model with the same training data and this time, the car drove much better in autonomous mode simulation.

All these experiments have in common that the models overfits with a small subset of image data. 

I then added dropout layers after the fully connected layers which helped in generalizing the problem and thereby reduce overfitting. 

#### Final Model

The first stage in my model includes preprocessing of data:

* Image cropping to focus on the road.
* Image normalization using per_image_standardization
 
The basic model from NVIDIA is as shown:

![Nvidial Nework Architecture][image_nvidia_model]

The model I trained has the following layers:


* [Convolutional Neuronal Network (CNN) Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) from NVIDIA
 * Convolutional layer with 2x2 stride and 5x5 Kernel
   * 5 x 5 Kernel, Output Depth 24, Subsampling 2 x 2, Activation Function: Elu
   * 5 x 5 Kernel, OUtput Depth 36, Subsampling 2 x 2, Activation Function: Elu
   * 5 x 5 Kernel, Output Depth 48, Subsampling 2 x 2, Activation Function: Elu
 * Convolutional layer without stride and 3x3 Kernel
   * 3 x 3 Kernel, Output Depth 64, Activation Function: Elu
   * 3 x 3 Kernel, Output Depth 64, Activation Function: Elu
 * Fully Connected Network
   * Flatten
   * Layer 100 Nodes
   * Dropout Layer
   * Layer 50 Nodes
   * Dropout Layer
   * Layer 10 Nodes
   * Layer 1 Nodes

To get a better overview of the trained network, I saved the model.summary from the network which is as shown below.

Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________



#### Network Parameters

While chosing the network parameters, I used the first guess with respect to the number of epochs. I chose a value of 10 in the beginning and plotted the network loss and validation loss against the number of epochs. I observed that although, after 10 epochs, the network loss decreased, but it still did not reach a saturated state. The validation loss had peaks in between. I tried with an increased batch size but the result was not better.

Then, I decided to increase the epochs to 20 and see if the losses converged. The network loss looked better this time with about to converge and the validation loss was varying. This could be due to the fact that I introduced dropout layers and shuffled the validation dataset before evaluation. Nonetheless, I could see that the car could drive smoothly in the simulator. The final network parameters I chose to train the model are mentioned below.

* Numer of epochs : 20
* Batch Size : 10
* Validation Split : 0.2
* Verbose : 1

The plot of network loss and validation loss against the number of epochs is as shown below.
![loss][loss_image]

### Result Video

This is an excerpt of the autonomous driving on track one. The resuting video file is named video.mp4

![Autonomous Driving][video_autonomous_driving]

