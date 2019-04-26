
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import get_source_inputs
from keras.utils import multi_gpu_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#TRAINING AND TESTING
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu')) #Convolution Layer 1
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu')) #Convolution Layer 2
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu')) #Fully connected layer 1
model.add(Dense(units=4, activation='softmax')) #Fully connected layer 2

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128,128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 800)


#Predction


image = image.load_img('prediction_Image/predict.jpg', target_size = (128,128))
image = image.img_to_array(image)
image = np.expand_dims(image, axis = 0)
result = model.predict(image)
training_set.class_indices
print(result.shape)
if result[0][0] == 1:
    gesture = 'FIST Detected'
elif result[0][1] ==1:
    gesture = 'OK Detected'
elif result[0][2] == 1:
    gesture = "PALM Detected"
else: gesture = 'THUMBS UP Detected'
print("Prediction Result :")
print(gesture)