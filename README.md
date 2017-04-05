#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_model]: ./img/model.png
[center]: ./img/center.jpg
[left]: ./img/left.jpg
[right]: ./img/right.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing the graph of the network
* model.h5 containing the weights of the above network
* mode.png an image of the network
* the images of the center camera used to train the final network 
* left track run using above training data
* right track run
* writeup_report.md this file

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing ```python drive.py model.json ```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The drive.py file contains the code for running the network. I is a slightly modified version of the drive.py as disgtributed. Main difference is the use of json for saving and loading the model.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The network was a variant of NVIDIA's with a number of modifications
* While the original NVIDIA had 9 layers (5 convolutional and 3 fully connected) we have 4 convolutional layers and 3 connected!
* Convolutional go:   24@5x5 -> 36@5x5 -> 48@5x5 -> 64@3x3. Then we flatten. Connected layers: 100 -> 50 -> 10 -> 1 (for steering angle)
* NOTE: As in current literature we used ELU (exponential linear unit) rather than RELU for activation
* The data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting.  Originally I had two but found the former counterproductive.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to empirically find a good network design.

My first step was to use a convolution neural network model similar to the one published by NVIDIA. I thought this model might be appropriate because it seemed to work well on the street and, unlike many other models, relied only on image data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. I found despite good numbers that the car was driving poorly (like a drunk) and worse still there were a few spots where the vehicle fell off the track.

One of the places where the car kept going off the road was the bridge in the left track. I recorded a large amount of data driving over the bridge. It did not help that much. The car seemed at most times a likely candidate for a DUI citation. 

I modified the network a number of times. I resized the images, I augmented the training data. I created more and more training data. I seemed that somethings got worse.

But despite the dispair at the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network modeled after NVIDIA's. 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image_model]


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][left]
![alt text][center]
![alt text][right]

(cameras left to right)


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to ....  Then I repeated this process in order to get more data points.

I even drove the car backwards on the track.

I tried to limit steering angle etc.

None of this I found really helped.

To augment the data sat, I also flipped images (and angles) when the steering angle was "large enough". 
My original idea was to try to get a random distribution of when to "flip".

This did not help either.

I tried training things for 20 some epochs but the car still drove like a drunkard.

I tried driving "like a drunkard" but this did not help. I found that using reasonable driving examples was important.

I readjusted the network. I threw away the idea of ramdonly choosing when to flip. I even went back to use the original dataset.

I ran only 3 epochs. And the car drove OK. It even drove OK on the 2nd track-- the left one-- which I had trouble trying to drive myself.

What I did discover is that the modi of the simulator had a great impact on the image quality. Higher quality images introduced a lot of shadows. The original training data and the images taken during runs seem to be lower quality without the shadows. A better approach it seems would have been to not just crop the image and normalize but to darken things and enhance the shadows to increase the robustness-- I decided to leave this when I have more time.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.



