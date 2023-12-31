import os
import pickle
from PIL import Image
import numpy as np

root = '/Data/federated_learning/data/cifar/data/'
imgs_path, labls_path = root+'test_X.npy',root+'test_y.npy'
labels = np.load(labls_path)
print('Loading images')
images = np.load(imgs_path)
images = images.reshape(labels.shape[0], 3,32,32)
print('y')