#importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

################ Data Preprocessing ###############

#processing the training set
train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

train_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64), batch_size = 32, class_mode='binary')

#preprocessing the test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')



############### Building the cnn ######################

#initialising the CNN
cnn = tf.keras.models.Sequential()

#convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu', input_shape=[64,64,3]))

#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#adding second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#full connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

#output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

############### Traning The CNN ###############

#compiling the cnn
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#training the CNN on the train set and evaluating on the test set
cnn.fit(x = train_set, validation_data = test_set, epochs = 25)

#making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)

train_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)