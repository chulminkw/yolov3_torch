import torch
from torch.utils.data import Dataset
import torchvision

import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2

class YoloDataset(Dataset):
    def __init__(self, df, transform, normalized_labels=True, verbose=False):
        self.image_files = df['img_path'].tolist()
        self.label_files = df['txt_anno_path'].tolist()
        self.max_objects = 100
        self.transform = transform
        self.normalized_labels = normalized_labels
        self.batch_count = 0
        self.verbose = verbose

    def __getitem__(self, index):
        image_path = self.image_files[index].rstrip()
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label_path = self.label_files[index]
        if os.path.exists(label_path):
            boxes_np = np.loadtxt(label_path).reshape(-1, 5)
            if self.verbose:
                print('image shape:', image.shape)
                print('boxes_np:', boxes_np)
            augmented = self.transform(image=image, 
                                       bboxes=boxes_np[:, 1:])
            aug_image = augmented['image']
            aug_boxes = torch.from_numpy(augmented['bboxes'])
            
            targets = torch.zeros(len(aug_boxes), 6)  
            targets[:, 2:] = aug_boxes
            targets[:, 1] = torch.from_numpy(boxes_np[:, 0])

        return aug_image, targets
    
    def __len__(self):
        return len(self.image_files)

# custom collate_fn 생성하여 batch 레벨로 출력값들이 동일한 shape가 되도록 변경. 
def custom_collate_fn(batch):
    images, targets = list(zip(*batch))
    # print(len(images), len(targets))
    # print(images[0].shape, images[1].shape, targets[0].shape, targets[1].shape)
    targets = [boxes for boxes in targets if boxes is not None]
    # Add sample index to targets
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i

    try:
        targets = torch.cat(targets, 0)
    except RuntimeError:
        targets = None  # No boxes for an image
    # print('targets after cat:', targets.shape)
    return images, targets
