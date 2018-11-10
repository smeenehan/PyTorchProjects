from glob import glob
import numpy as np
from os import path
from PIL import Image
import torch
import torch.utils.data as data
import torchvision
import xml.etree.ElementTree as et

VOC_ROOT = path.join(path.expanduser('~'), 'Documents', 'PASCAL', 'VOC2012')
VOC_CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
SET_FILES = {'train': 'train.txt', 'val': 'val.txt', 'merged': 'trainval.txt'}
VOC_MEAN = [0.4824, 0.4588, 0.4078]
MAX_NUM_OBJ = 56

class VOCDataset(data.Dataset):
    """ PyTorch Dataset for PASCAL VOC data. Returns tuples of image tensors and 
    target info, where targets are specified as normalized bbox coordinates and 
    class ids, in (y1, x1, y2, x2, id) format. To facilitate batching, the target
    array is always zero padded to have the mximum number of rows (MAX_NUM_OBJ).
    """
    def __init__(self, root_path=VOC_ROOT, set_type='merged', transform=None):
        self.image_paths, self.annotation_paths = get_image_paths(
            root_path=root_path, set_type=set_type)
        self.transform = transform

    def __getitem__(self, index):
        img, target = load_image_data(self.image_paths[index], 
                                      self.annotation_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        num_diff = MAX_NUM_OBJ-target.shape[0]
        if num_diff>0:
            target = np.pad(target, ((0, num_diff), (0, 0)), 'constant')
        return img, target

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(root_path=VOC_ROOT, set_type='merged'):
    """
    Parameters
    ----------
    root_path : string, optional
        Path containing VOC subfolders ('ImageSets', 'JPEGImages', and 'Annotations')
    set_type : string, optional
        Whether to return training set ('train'), validation set ('val'), or both ('merged')

    Returns
    -------
    tuple (list, list)
        Paths of all images in the selected set, paths of all corresponding annotation
        files
    """
    image_paths, annotation_paths = [], []
    image_name_file = path.join(root_path, 'ImageSets', 'Main', SET_FILES[set_type])
    with open(image_name_file, 'r') as file:
        image_names = file.read().splitlines()
    for name in image_names:
        image_paths.append(path.join(root_path, 'JPEGImages', name+'.jpg'))
        annotation_paths.append(path.join(root_path, 'Annotations', name+'.xml'))
    return image_paths, annotation_paths

def load_image_data(image_path, annotation_path):
    """
    Parameters
    ----------
    image_path : string
        Path of image JPEG
    annotation_path : string
        Path of annotation XML file containing class and bounding box info.

    Returns
    -------
    tuple (PIL Image, list of lists)
        Image and list of detection targets: normalized bounding box coordinates
        plus class id, in (y1, x1, y2, x2, ID).
    """
    image = Image.open(image_path).convert('RGB')
    target = _parse_annotation(annotation_path)
    return image, target

def _parse_annotation(annotation_path):
    root = et.parse(annotation_path).getroot()
    #class_ids, class_labels, bboxes = [], [], []
    targets = []
    size_info = root.find('size')
    height = int(size_info.find('height').text)
    width = int(size_info.find('width').text)

    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        class_id = VOC_CLASSES.index(obj_name)
        bbox_info = obj.find('bndbox')
        y1, y2 = int(bbox_info.find('ymin').text), int(bbox_info.find('ymax').text) 
        x1, x2 = int(bbox_info.find('xmin').text), int(bbox_info.find('xmax').text)
        target = [y1/height, x1/width, y2/height, x2/width, class_id]
        targets.append(target)
    return np.array(targets)

