import os.path
from typing import Any, Callable, cast, Optional, Tuple

import numpy as np
from PIL import Image

#from .utils import check_integrity, download_and_extract_archive, verify_str_arg
#from .vision import VisionDataset


class STL10():




    def __init__(
        self,
        root,
        split= "train"
    ):
        self.class_names_file = "class_names.txt"
        self.train_list = [
            ["train_X.npy"],
            ["train_y.npy"],
            ["unlabeled_X.npy"],
        ]

        self.test_list = [["test_X.npy"], ["test_y.npy"]]
        self.splits = ("train", "train+unlabeled", "unlabeled", "test")
        self.root = root
        self.split = split
        # now load the picked numpy arrays

        if self.split == "train":
            self.data, self.labels = self.loadfile(self.train_list[0][0], self.train_list[1][0])


        elif self.split == "train+unlabeled":
            self.data, self.labels = self.loadfile(self.train_list[0][0], self.train_list[1][0])
            unlabeled_data, _ = self.loadfile(self.train_list[2][0])


        elif self.split == "unlabeled":
            self.data, _ = self.loadfile(self.train_list[2][0])
        else:  # self.split == 'test':
            self.data, self.labels = self.loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))


        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]



    def loadfile(self, data_file, labels_file= None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.load(f)  # 0-based

        path_to_data = os.path.join(self.root, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.load(f)
            images = np.reshape(everything, (-1, 3, 32, 32))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels
