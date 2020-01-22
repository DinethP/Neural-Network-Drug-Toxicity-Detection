#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import csv


# In[ ]:


test_data = np.genfromtxt("../Tox21_AR/score_graphs.csv", delimiter=',')

test_data = test_data.reshape((test_data.shape[0] // test_data.shape[1], test_data.shape[1], test_data.shape[1], 1))


# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=test_data.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


# In[ ]:

checkpoint_path = "cp.ckpt"
model.load_weights(checkpoint_path)


predictions = model.predict_classes(test_data, 2)  # predict the score_graphs using the model


# In[ ]:

np.savetxt('labels.txt', predictions, fmt='%1.0f')

