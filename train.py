#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

import numpy as np
import tensorflow as tf


# In[3]:


with open("./train_data/train_graphs.csv", "r") as data:
    file = csv.reader(data, delimiter=',')
    train_data = [r for r in file]
with open("./train_data/train_labels.csv", "r") as labels:
    file = csv.reader(labels, delimiter=',')
    train_labels = [r for r in file]
train_labels = [r[1] for r in train_labels]
train_data = np.asarray(train_data, dtype=int)
train_labels = np.asarray(train_labels[1:], dtype=int)
train_data = train_data.reshape((train_data.shape[0] // train_data.shape[1], train_data.shape[1], train_data.shape[1], 1))


# In[8]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=train_data.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


# In[9]:


checkpoint_path = "cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


# In[ ]:


model.fit(train_data, train_labels, epochs=5, callbacks=[cp_callback]) # train the model
