
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim

df = pd.read_csv("/content/mnist_train.csv")
df = df[:1000]

df.head()

train_labels = np.array(df['label'])
train_labels = np.reshape(train_labels, (train_labels.shape[0], 1))

train_labels.shape

train_images = []
for i in range(len(df)):
  col = df.iloc[i]
  pixel_vals = col[1:].to_list()
  train_images.append(pixel_vals)

train_images = np.array(train_images)

train_images = train_images / 255.0

CODING_DIM = 10

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # Encoder Block
    self.e1 = nn.Linear(784, 128)
    self.e2 = nn.ReLU()
    self.e3 = nn.Linear(128, 64)
    self.e4 = nn.ReLU()
    self.e5 = nn.Linear(64, CODING_DIM)

    # Decoder Block
    self.d1 = nn.Linear(CODING_DIM, 64)
    self.d2 = nn.ReLU()
    self.d3 = nn.Linear(64, 128)
    self.d4 = nn.ReLU()
    self.d5 = nn.Linear(128, 784)
    self.d6 = nn.Sigmoid()

  def forward(self, x):
    x = self.e1(x)
    x = self.e2(x)
    x = self.e3(x)
    x = self.e4(x)
    x = self.e5(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    return x

net = Net()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
EPOCHS = 10
for epoch in range(EPOCHS):
  running_loss = 0
  for i in range(len(train_images)):
    optimizer.zero_grad()
    inputs = torch.Tensor(train_images[i])
    outputs = net(inputs)
    loss = loss_fn(outputs, inputs)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')

print('Finished Training')

outputs = net(torch.Tensor(train_images[1]))
outputs = torch.reshape(outputs, (28, 28))

plt.imshow(outputs.detach().numpy(), cmap='binary')
plt.title(f"Actual Number: {train_labels[1][0]}")
plt.show()

import random

for i in range(10):
  random_ind = random.randint(0, len(df))
  inputs = torch.Tensor(train_images[random_ind])
  outputs = net(inputs)
  outputs = torch.reshape(outputs, (28, 28))
  plt.imshow(outputs.detach().numpy(), cmap='binary')
  plt.title(f"Actual Number: {train_labels[random_ind][0]}")
  plt.show()

