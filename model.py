# this is a scipt for create and train the model
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
# load the data
## use all of  direction data
#current_data_path = './selfData/'
lines = []
## read csv file
with open("./selfData/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    isfirst = True
    for line in reader:
        # del the first line which is a column name
        if isfirst:
            isfirst =False
            continue
        lines.append(line)

## read image and steering_angle
### flip data
images = []
measurements  =[]
bias_steering = 0.3
index =0
for line in lines:
    current_steering_center = float(line[3])
    current_steering_left = current_steering_center + bias_steering
    current_steering_right = current_steering_center - bias_steering
    image_path = line[0]
    #current_path = current_data_path + line[0]
    img_center = cv2.imread(line[0])
    #img_left = cv2.imread(current_data_path + line[1])
    #img_right = cv2.imread(current_data_path + line[2])

    images.append(img_center)
    #images.append(img_left)
    #images.append(img_right)
    measurements.append(current_steering_center)
    #index += 1
    #if index >5:
    #    break
    #measurements.append(current_steering_left)
    #measurements.append(current_steering_right)

    #images.append(cv2.flip(img_center,1))
    #images.append(cv2.flip(img_left,1))
    #images.append(cv2.flip(img_right,1))

    #measurements.append(-current_steering_center)
    #measurements.append(-current_steering_left)
    #measurements.append(-current_steering_right)
for i in range (len(images)):
    images.append(cv2.flip(images[i],1))
    measurements.append(-measurements[i])
#### test read 
#print(measurements[0])
#plt.imshow(images[0])
#plt.show()
#plt.show()
## test cropping region
image_top = 50
image_bottom = images[0].shape[0] - 20
#region = images[12]
#region = region[image_top:image_bottom,:]
#plt.imshow(region)
#plt.show()

## create the net architecture
X_train= np.array(images)
#print(X_train.shape)
#print(X_train[0].shape)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Lambda,MaxPooling2D,Dropout
from keras.layers.convolutional import Cropping2D,Convolution2D
#print(X_train[12].shape)

model = Sequential()
#model.add(Flatten(input_shape = (160,320,3) ))
## preprecessiong data
model.add(Cropping2D(cropping=((50,30),(0,0)),input_shape = (160,320,3)))
#model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape = (80,320,3)))
# test net
model.add(Convolution2D(6,3,3,border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16,3,3,border_mode= 'valid'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

# output shape is 320,80
#model.add(Convolution2D(36,5,5,border_mode = 'valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((3,3)))
# output shape is 24 31 ,98
#model.add(Convolution2D(36,5,5,border_mode= 'valid',input_shape = (160,320,3)))
#model.add(Activation('relu'))
# output shape is 36 98 31
#model.add(Convolution2D(64,5,5,border_mode= 'valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((5,5)))
# output shape is
#model.add(Convolution2D(16,3,3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((3,3)))
# flatten
#model.add(Flatten())
#model.add(Activation('relu'))
#model.add(Activation('relu'))
#model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
#model.add(Dense(1))
#model.add(Activation('softmax'))
## concern the optimizer concern the loss
model.compile(loss = 'mse',optimizer = 'adam')
## train 
fit = model.fit(X_train,y_train,validation_split=0.2,shuffle = True,nb_epoch = 1)
## visul the data

## save the data as model.h5
model.save('model.h5')
