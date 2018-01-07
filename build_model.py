#====================================================================
# Project 3 in Udactiy Self Driving Car Nano Degree 
# Using a Convolutional Neural Network to Predict Steering Angles
# syntax upgraded to keras 2 to make keras-vis possible
#====================================================================

#====================================================================
# import libraries 
# for data read in and CNN
import csv 
import cv2
import numpy  as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, MaxPooling2D, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 	
from keras.regularizers import l2
from keras.optimizers import Adam 

# for balancing the data by steering angle
import matplotlib.pyplot as plt
import pylab
import numpy.random as random

# image data generator and train test split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#====================================================================

#====================================================================
# Read in training data 
# from Udacity class material, reads in y
lines = []
with open('./data/train3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
	
# from Udacity class material, reads in X
images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		# split character modified because image data was collected on windows 10
		filename = source_path.split('\\')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.imread(current_path)
		# add extra cropping
		image_c = image[65:140,0:320,:]
		#masked_image = image_mask(image_c)
		images.append(image_c)
		measurement = float(line[3])
		measurements.append(measurement)

# from Udacity class material, doubles X by making a mirror image of each
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)
#====================================================================

	
#====================================================================
# further processing by balancing steering angle values to get a more
#   uniform distribution

# produce random selection 
# idea from https://stackoverflow.com/questions/19485641/python-random-sample-of-two-arrays-but-matching-indices

# defines upsampling based on steering angle value
def notch_calc(x, y_int):
    y = (x)**2 + y_int
    return y

# Resamples data set accroding to total number and notch_calc values 
def balance_steering(X, y, num_bins, len_multiplier, notch = False, notch_multiplier = 3):
    y_trim = []
    X_trim = []
    bal = np.linspace(-1, 1, num_bins)
    calc_hist = plt.hist(y, bins = bal)
    tot = sum(calc_hist[0])
    dist = calc_hist[0]
    non_z = dist[dist !=0]
    #max_size = np.round((len(y) * len_multiplier) / num_bins)
    even_size = ((tot*len_multiplier) / len(non_z))
    #inverse_dist = even_size - dist 
    #inverse_dist[inverse_dist <= 0] = even_size
    samp_size = np.repeat(even_size, len(dist))
    samp_size = even_size.astype(int)
    if notch == True:
        #rep_even = np.repeat(even_size, len(dist))
        notch_size = notch_calc(np.linspace(-1, 1, len(dist)), y_int = (1/notch_multiplier))
        notch_sample = notch_size * even_size
        samp_size = notch_sample.astype(int)
    # start for loop, should return pretty balanced X a d y
    for i in range(0, (len(bal)-1)):
        y_trim_e = [y for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]
        X_trim_e = [X for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]
        if len(y_trim_e) != 0:
            random_index = random.choice(a = range(len(y_trim_e)), size = samp_size[i], replace = True)
            y_trim += [y_trim_e[i] for i in random_index]
            X_trim += [X_trim_e[i] for i in random_index]
    return X_trim, y_trim
    
    
# run the balance
X_trim, y_trim = balance_steering(augmented_images, augmented_measurements, num_bins = 50, len_multiplier = .1, notch = True, notch_multiplier = .2)
#====================================================================


#====================================================================
# visualize distributions, comment out if running whole script at once
# take another look
#bal = np.linspace(-1, 1, 50)

#bal = np.linspace(-1, 1, 20)

#plt.hist(y_trim, bins = bal) 
#pylab.show()

# end vis
#====================================================================

#====================================================================
# put training data in np arrays to feed into CNN
X_train = np.array(X_trim)
y_train = np.array(y_trim)
#====================================================================

#====================================================================
# now split into train and valid 
X_valid, X_train, y_valid, y_train = train_test_split(X_train, y_train, test_size=0.8, shuffle = True, random_state=9323)
#====================================================================


#====================================================================
# Keras !!2.0!! CNN. 5 convolutional layers and 4 fully connected with dropout. 
# additionally, data is normalized and cropped in the initial layers. 
# multiple dropout and l2 regularizations added. Additionally, only one
#   layer in the fully connected section is given an activiation function. 
#  In practice, it was found no activations in fully connceted layers 
#  produced models which drove "twitchier" and were better at recognizing recovery. 
#  However, they were prone to over correction and crashes around difficult corners. 
#  Adding one activation function sandwiched between to fully connected linear layers 
#   provided a more ideal amount of response. 
# Trained with MSE as loss function and adam optimizer from Keras. 
#====================================================================

#====================================================================
# build some custom functions. 
# random dark spots
 #  inspired by jermomy shannon

def random_spot(img):
    new_img = img.astype(float)
    flip = np.random.choice(2, 1)
    if flip == 1:
        rounds = np.random.randint(1, 5)
        for i in range(0, rounds):
            value = np.random.randint(-110, 110)
            c_1 = np.random.randint(0, 320)
            c_2 = np.random.randint(0, 320)
            c_3 = c_1 + c_2
            if c_3 > 320: 
                c_3 = 320
            r_1 = np.random.randint(0, 75)
            r_2 = np.random.randint(0, 75)
            r_3 = r_1 + r_2
            if r_3 > 75:
                r_3 = 75
            mask = new_img[r_1:r_3, c_1:c_3,:] - value
            mask[mask < 0] = 0
            new_img[r_1:r_3, c_1:c_3:] =  mask 
    return new_img
#====================================================================

#====================================================================
# image data generator: 
# source[https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html]

# training generator
datagen_train = ImageDataGenerator(
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
        #adaptive_equalization = False, 
    rotation_range = 15,
    width_shift_range=0,
    channel_shift_range = 35,
    shear_range = 0, 
    preprocessing_function = random_spot,
        #blurring = 3,
        #zca_whitening = True, 
        #zca_epsilon = 1e-6,
        #height_shift_range=0.1,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.1,
        #horizontal_flip=True,
    fill_mode='constant')
    
        
# validation generator      
datagen_valid = ImageDataGenerator(
    featurewise_std_normalization = False,
        #adaptive_equalization = False, 
    samplewise_std_normalization = False, 
    rotation_range=20,
    width_shift_range=0.05,
    channel_shift_range = 45,
    preprocessing_function = random_spot,
        #blurring = 3,
    shear_range = 0, 
        #zca_whitening = True, 
        #zca_epsilon = 1e-6,
        #height_shift_range=0.1,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.1,
        #horizontal_flip=True,
    fill_mode='constant')

# fit to image groups        
datagen_train.fit(X_train)
datagen_valid.fit(X_valid)
#datagen_valid.fit(X_train)
#=======================================================================


#=======================================================================
# set up the flow method !! checks the method. 
# comment out when running whole script

# use the generator object to print training data examples
#X_valid_b, y_valid_b = next(datagen_valid.flow(X_valid, y_valid, batch_size=20))

# take a look at some pics
#for i in range(0,20):
#    cv2.imwrite('test_' + str(i) + '.jpg', X_valid_b[i])
    
#=======================================================================


#=======================================================================
# set up the flow method !! sets up for fit generator
train_generator = datagen_valid.flow(X_train, y_train, batch_size=32)
valid_generator = datagen_valid.flow(X_valid, y_valid, batch_size=32)
#valid_generator = datagen_valid.flow(X_train, y_train, batch_size=64)
#=======================================================================


#=======================================================================
# begin model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (75,320,3)))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(48, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(64, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(64, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer = l2(.0001), activation = 'elu'))
model.add(Dense(25,  kernel_regularizer = l2(.0001), activation = 'elu'))
model.add(Dense(10, kernel_regularizer = l2(.0001), activation = "elu"))
model.add(Dense(1))
adam = Adam(lr = 0.0001)
model.compile(optimizer = adam, loss='mse')
# end model 
#====================================================================


#====================================================================
# train model and save output
# run model with fit generator 
model.fit_generator(
        train_generator,
        #steps_per_epoch= 200,
        samples_per_epoch = 5000, 
        epochs=20,
        validation_data = valid_generator,
        validation_steps = 50)

model.save('model.h5')

exit()
#====================================================================
