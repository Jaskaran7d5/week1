#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import cv2
from torch.utils.data import DataLoader, TensorDataset


# In[7]:


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
#We use this class to create our dataset for training
class DeblurDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.blurred_images = sorted(os.listdir(blurred_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))
    
    def __len__(self):
        return len(self.blurred_images)
    #
    def __getitem__(self, idx):
        blurred_image = Image.open(os.path.join(self.blurred_dir, self.blurred_images[idx])).convert('RGB')
        sharp_image = Image.open(os.path.join(self.sharp_dir, self.sharp_images[idx])).convert('RGB')
        
        if self.transform:
            blurred_image = self.transform(blurred_image)
            sharp_image = self.transform(sharp_image)
        
        return blurred_image, sharp_image

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create the dataset
blurred_dir = r"C:\users\jaska\blur_photu\defocused_blurred"
sharp_dir = r"C:\users\jaska\sharp"
dataset = DeblurDataset(blurred_dir, sharp_dir, transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
#Creating a class unet, for making a unet for image deblurring
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoding path
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.output_layer(d1)

# Instantiate the model
model = UNet(in_channels=3, out_channels=3)


# In[9]:


import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (blurred_images, sharp_images) in enumerate(dataloader):
        blurred_images = blurred_images.to(device)
        sharp_images = sharp_images.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(blurred_images)
        loss = criterion(outputs, sharp_images)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / (i + 1):.4f}')
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')


# In[13]:


#saving the model so that it should not be trained again and again

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

