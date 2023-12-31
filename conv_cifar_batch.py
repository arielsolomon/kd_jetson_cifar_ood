import os
import pickle
from PIL import Image
import numpy as np
from time import time
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def split_arr(images, labels, dest):

    num_items = len(labels)
    frac_train, frac_test, frac_unlabeled = 0.15,0.05,0.8
    arr_train_img, arr_test_img, arr_unlabeled_img = images[:int(frac_train*num_items),:,:,:],\
        images[int(frac_train*num_items):int(frac_train*num_items)+int(frac_test*num_items),:,:,:],\
        images[int(frac_train*num_items)+int(frac_test*num_items):,:,:,:]
    np.save(dest+'train_X.npy',arr_train_img,allow_pickle=True),\
        np.save(dest+'test_X.npy',arr_test_img,allow_pickle=True),\
        np.save(dest+'unlabeled_X.npy',arr_unlabeled_img,allow_pickle=True)

    arr_train_lbl, arr_test_lbl, arr_unlabeled_lbl = labels[:int(frac_train * num_items)], \
        labels[int(frac_train * num_items):int(frac_train * num_items) + int(frac_test * num_items)], \
        labels[int(frac_train * num_items) + int(frac_test * num_items):]
    np.save(dest + 'train_y.npy', arr_train_lbl,allow_pickle=True), np.save(dest + 'test_y.npy', arr_test_lbl,allow_pickle=True), np.save(
        dest + 'unlabeled_y.npy', arr_unlabeled_lbl,allow_pickle=True)

dest = '/Data/federated_learning/data/cifar/data/'
if not os.path.exists(dest):
    os.mkdir(dest)
root = '/Data/federated_learning/data/cifar/cifar-10-batches-py/'
counter = 0
images, labels = np.empty((0, 3, 32, 32)), np.empty((0,), dtype=int)  # Ensure integer dtype for labels
for file in os.listdir(root):
    print('started batch', file)
    start = time()
    path = root+file
    file_dict = unpickle(path)
    count = 0
    for image, label in zip(file_dict[b'data'], file_dict[b'labels']):
        image = image.astype(np.uint8)
        images = np.concatenate([images, [image.reshape(3,32,32)]])
        labels = np.append(labels, label)
        labels = labels.astype(np.int64)
    print('eneded batch number: ', file)

split_arr(images, labels, dest)



