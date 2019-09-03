from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as func

VOC_ROOT = Path.home() / 'Documents' / 'PASCAL'
VOC_CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
VOC_MEAN = [0.4824, 0.4588, 0.4078]

def extract_VOC_detections(raw_target):
    """Given a VOC target, as an XML tree, extract an array of all bounding boxes
    and class IDs, each in the format (y1, x1, y2, x2, ID), where bounding box
    coordinates are normalized
    """
    targets = []
    annotation = raw_target['annotation']
    height = int(annotation['size']['height'])
    width = int(annotation['size']['width'])
    objects = annotation['object']
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        obj_name = obj['name']
        class_id = VOC_CLASSES.index(obj_name)
        bbox_info = obj['bndbox']
        y1, y2 = int(bbox_info['ymin']), int(bbox_info['ymax']) 
        x1, x2 = int(bbox_info['xmin']), int(bbox_info['xmax'])
        annotation = [y1/height, x1/width, y2/height, x2/width, class_id]
        targets.append(annotation)
    return torch.tensor(targets)

def collate_VOC_batch(batch):
    """Collate a batch of VOC detection samples, provided as a list of (img, targets)
    tuples. The first element img is assumed to be a valid PyTorch Tensor, and all 
    images should be the same size: this should be handled in the Dataset transform.
    The Tensor of targets (normalized bounding boxes), on the other hand, is variable from
    image to image, so we will zero pad appropriately when collating
    """
    images = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    max_num_targets = np.max([t.shape[0] for t in targets])
    padded_targets = []
    for t in targets:
        num_diff = max_num_targets-t.shape[0]
        if num_diff>0:
            t = func.pad(t, (0, 0, 0, num_diff), 'constant')
        padded_targets.append(t)
    return torch.stack(images), torch.stack(padded_targets)

