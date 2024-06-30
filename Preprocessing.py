'''Project: SliceGAN
   Part: Preprocessing
   Author: lwj
   Date: 2024/2/21-2024/2/25'''

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

# Import original pictures from file
pic_list = []
i = 0
for pic_name in os.listdir('C:/Users/Bitte/Downloads/sofc/anode_segmented'):
    file_name = 'C:/Users/Bitte/Downloads/sofc/anode_segmented' + '/' + pic_name
    img = Image.open(file_name)
    img = img.convert('RGB')
    # print(img.size)
    i += 1
    if (i - 1) % 50 == 0 :
        pic_list.append(img)

# Transform pictures into nparray
pic_data_list = []
for item in pic_list:
    img_data = np.array(item)
    pic_data_list.append(img_data)
    # print(pic_data_list[0])
size = pic_data_list[0].shape
    
# Crop images(data)
divipic_data_list = []
for image in pic_data_list :
   for i in range(1 , size[0] , 8) :
        for j in range(1 , size[1] , 8) :
            if (i + 64) < size[0] and (j + 64) < size[1] and len(divipic_data_list) < 140288:
                roi = image[i : (i + 64) , j : (j + 64)]
                divipic_data_list.append(roi)
# print(divipic_data_list[0])
# print(divipic_data_list[1])
# pic_shape = divipic_data_list[0].shape
# print(pic_shape)
plt.imshow(divipic_data_list[1])

# One-hot encoding
onehot_pic_list = []

for pic in divipic_data_list:
    onehot_pic = np.transpose(pic , (2,0,1))
    onehot_pic_list.append(onehot_pic)

onehot_data = np.array(onehot_pic_list)
sample = np.unique(onehot_pic_list[0])
sample = sample.reshape(1 , -1 , 1 , 1)
channel_data = np.where(onehot_data == sample , 1 , 0)

data = torch.FloatTensor(channel_data)
dataset = torch.utils.data.TensorDataset(data)
print(data.shape)