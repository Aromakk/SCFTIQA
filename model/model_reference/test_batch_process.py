import cv2
import os
import numpy as np
import torch
from model.saliency_process import patch_process
from PIL import Image
import matplotlib.pyplot as plt


image_folder = 'E:data\LIVE_challenge\image_test'
saliency_map_folder = 'E:data\LIVE_challenge\mask_test'

image_file_list = os.listdir(image_folder)
saliency_map_file_list = os.listdir(saliency_map_folder)

B = len(image_file_list)

batch_images = []
batch_saliency_maps = []

for i in range(B):
    image_path = os.path.join(image_folder, image_file_list[i])
    saliency_map_path = os.path.join(saliency_map_folder, saliency_map_file_list[i])

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :, :, :]

    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.resize(saliency_map, (224, 224))
    saliency_map = saliency_map[np.newaxis, np.newaxis, :, :]
    print("saliency_map",saliency_map.shape)


    image_show = torch.tensor(image)
    print(image_show.shape)
    print(type(image_show))
    saliency_map_show = torch.tensor(saliency_map)

    result = patch_process(image, saliency_map)
    image_show = torch.squeeze(image_show)
    image_show = image_show.permute(1,2,0)

    saliency_map_show = torch.squeeze(saliency_map_show)
    print(saliency_map_show.shape)

    plt.imshow(saliency_map_show, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(image_show)
    plt.axis('off')
    plt.show()

    batch_images.append(image)
    batch_saliency_maps.append(saliency_map)

