# **Behavioral Cloning**

The output video can be seen [here](https://www.youtube.com/watch?v=yidhNS1pF3w&t=16s).

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers 4 fully connected layers (this is the NVIDIA end-to-end learning model). The first three convolutional layers use a filter of size 5x5, a stride of 2x2 and depths of 24, 36 and 48. The last two convolutional layers use a filter of size 3x3, no stride and depths of 64 and 64. The fully connected layers are of sizes 1164, 100, 50 and 10. 

The model includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer. The incoming image is cropped from the top and bottom by 60 and 25 pixels respectively using a Keras Cropping2D layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and recorded the data for multiple laps.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out different pre-existing network architectures, analyse their performances and then modify them accordingly.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it has worked in the past and well was the first convolutional network ever so why not start with this. When I ran this model on my dataset I found that the network wasn't able to perform well which was expected and thus I decided to increase the depth of the network by increasing the number of convolutional and fully connected layers. The next best option was to use the NVIDIA end-to-end learning model as it has proved to work in the past for the same application.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more data for those particular areas and how to recover if going off track. This worked out pretty well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track if it goes off. I then turned the car around and repeated the same process while driving in the opposite direction. This was done as the first track has a left bias so in order to generalize the data I recorded data in the opposite direction too.

Then I repeated this process on track two in order to get more data points and make it more generic.

To augment the data sat, I also flipped images thinking that this would help the model learn better and not just learn this particular track.

After the collection process, I had 91296 number of data points. I then preprocessed this data by normalizing it and cropping the bottom 25 and top 60 pixels. The cropping was done in order to avoid looking at data that is not important (bottom: car hood, top: sky and trees)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
