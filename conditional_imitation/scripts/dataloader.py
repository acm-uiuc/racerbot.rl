from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
import torchvision.transforms as transforms
import cv2

def read_video(filename):
    path = '/home/rohang62/Documents/racerbot.rl/conditional_imitation/scripts/frames'
    dataset = []
    vid = cv2.VideoCapture(filename)
    sucess = True
    success,image = vid.read()
    count = 1
    while success:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.imshow("image", grayscale)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(path, "frame%d.jpg" % count), grayscale)
        count += 1
        success,image = vid.read()

    return path, count - 1

class AVData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.dir, self.len = read_video(dataset)
        self.steering = np.random.randint(1,101,len(self))
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        s = cv2.imread(os.path.join(self.dir, "frame%d.jpg" % index))
        return self.transform(s), self.steering[index - 1]

# dataset = AVData(read_video("vid.mp4"))
# train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
# imgs, steering = next(iter(train_loader))
# print('Batch shape:',imgs.numpy().shape)
# for i in range(imgs.shape[0]):
#     plt.imshow(imgs.numpy()[i,:,:,:])
#     plt.show()
