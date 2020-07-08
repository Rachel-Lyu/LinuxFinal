#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.image as mp
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    hidden1 = 512
    hidden2 = 512
    self.fc1 = nn.Linear(28 * 28, hidden1)
    self.fc2 = nn.Linear(hidden1, hidden2)
    self.fc3 = nn.Linear(hidden2, 10)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = x.view(-1, 28*28)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    
    x = F.relu(self.fc2(x))
    x = self.dropout(x)

    x = self.fc3(x)
    return(x)

model = Net()
print(model)
model.load_state_dict(torch.load('model.pt'))


# In[39]:


# img = mp.imread("./examples/numbers/1.jpg")


# In[40]:


def num_predict(img):
    img.reshape((28, 28))
    img = img / img.max()
    img = torch.tensor(img).float()
    result = F.softmax(model(img), dim=1)
    result = result.detach().numpy().reshape(10)
    return result


# In[41]:


# num_predict(img)

