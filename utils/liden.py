#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as mp
from keras.models import load_model
model = load_model('./models/model.kmd')


# In[2]:


def pred_letter(img):
    img = img / img.max()
    img = img.reshape((1, 28, 28, 1))
    return model(img).numpy().flatten()


# In[3]:


# img = mp.imread('./examples/letters/1.jpg')
# pred_letter(img)

