#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:36:28 2020

@author: mathew
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
# Directory with our training data containing cats and dogs
train_images = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/train/')
# Directory with our test data containing cats and dogs
test_images = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/validation/')

dog_train = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/train/dogs')
cat_train = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/train/cats')

dog_test = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/validation/dogs')
cat_test = os.path.join('/home/mathew/Documents/Mathew/Cats-vs-Dogs-Classification-Model/validation/cats')


print('Total training images_dogs:', len(os.listdir(dog_train)))
print('Total training images_cats:', len(os.listdir(cat_train)))

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
#%%
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    # This is the source directory for training images
    train_images,
    target_size=(150, 150),  # All images will be resized to 150x150
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
test_generator = test_datagen.flow_from_directory(
    # This is the source directory for training images
    test_images,
    target_size=(150, 150),  # All images will be resized to 150x150
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
#%%
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    # Dropout layer to prevent overfitting
    # tf.keras.layers.Dropout(.3),
    # outputs to the 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.optimizer.lr = .001

history = model.fit(train_generator, epochs=10, 
                    validation_data=test_generator, verbose=1)

#%%
model.save('Final_Model.h5')

# %%
