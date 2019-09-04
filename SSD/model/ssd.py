from itertools import product
from math import ceil, floor

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16

MIN_SCALE = 0.2
MAX_SCALE = 0.9
ASPECTS = (2, 0.5, 3, 1/3)

class SSD(nn.Module):
    """Implement an SSD-style object detector.

    Run input images through a backbone network (e.g., VGG, ResNet, with no classifier),
    then a set of extra feature layers. Multiple source layers in the backbone and
    extra networks are then run through convolutional filters to make multi-scale
    bounding-box and class predictions.

    Parameters
    ----------
    input_shape : tuple
        Shape of the inputs, in CHW format.
    backbone : list of torch.nn.Module
        Module layers representing the backbone classifier. The output
        of each element will be used as sources for detection.
    num_classes : int
        Number of output classes (should include a background class).
    """
    def __init__(self, input_shape, backbone, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = nn.ModuleList(backbone)

        """Head network which will convert the multiple sources to per-pixel sets
        of bounding box offsets and class predictions. We will build this as we go, since
        it's most easily determined by tracing a dummy tensor through the layers"""
        self.loc_layers = nn.ModuleList()
        self.class_layers = nn.ModuleList()

        """Trace through the backbone, adding the appropriate head layers and priors as necessary,
        and log the final shape so we can properly build the extra layers"""
        map_sizes = []
        num_bb_boxes = [6]*len(self.backbone)
        num_bb_boxes[0] = 4
        with torch.no_grad():
            x = torch.zeros(self.input_shape)[None, :]
            for num_boxes, layer in zip(num_bb_boxes, self.backbone):
                layer.eval()
                x = layer(x)
                C_out = x.shape[1]
                map_sizes.append(x.shape[2])
                self.loc_layers.append(
                    nn.Conv2d(C_out, num_boxes*4, 3, padding=1)
                )
                self.class_layers.append(
                    nn.Conv2d(C_out, num_boxes*self.num_classes, 3, padding=1)
                )
        backbone_shape = x.size()[1:]

        """Build extra layers, and continue to trace through and add head layers"""
        self.extra = nn.ModuleList(build_extra(backbone_shape))
        num_extra_boxes = [6]*len(self.extra)
        num_extra_boxes[-2:] = [4, 4]
        with torch.no_grad():
            for num_boxes, layer in zip(num_extra_boxes, self.extra):
                layer.eval()
                x = layer(x)
                C_out = x.shape[1]
                map_sizes.append(x.shape[2])
                self.loc_layers.append(
                    nn.Conv2d(C_out, num_boxes*4, 3, padding=1)
                )
                self.class_layers.append(
                    nn.Conv2d(C_out, num_boxes*self.num_classes, 3, padding=1)
                )
        
        all_boxes = num_bb_boxes+num_extra_boxes
        num_sources = len(all_boxes)
        defaults = []
        for idx, num_boxes in enumerate(all_boxes):
            defaults += default_boxes(input_shape[1], map_sizes[idx], 
                                      idx, num_sources, num_boxes)
        self.default_boxes = torch.tensor(defaults).view(-1, 4)
        self.default_boxes.clamp_(max=1, min=0)

    def forward(self, x):
        sources = []
        for layer in self.backbone:
            x = layer(x)
            sources.append(x)

        for layer in self.extra:
            x = layer(x)
            sources.append(x)

        locs, classes = [], []
        for (x, l, c) in zip(sources, self.loc_layers, self.class_layers):
            # Note the call to contiguous, so that we can use view() to reshape w/out copying
            locs.append(l(x).permute(0, 2, 3, 1).contiguous())
            classes.append(c(x).permute(0, 2, 3, 1).contiguous())

        locs = torch.cat([x.view(x.size(0), -1) for x in locs], 1)
        classes = torch.cat([x.view(x.size(0), -1) for x in classes], 1)
        if self.training:
            output = (locs.view(locs.size(0), -1, 4),
                      classes.view(classes.size(0), -1, self.num_classes),
                      self.default_boxes)
        else:
            output = None
        return output


def build_extra(input_shape):
    """Extra layers after the backbone, split into multiple Sequential
    segments to get sources from intermediate layers"""
    C, H, W = input_shape
    if H != W:
        raise ValueError('Only square inputs supported.')
    C_in, C_mid, C_out = C, C//4, C//2

    extra =[]
    while H > 5:
        layers = [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                  nn.Conv2d(C_mid, C_out, 3, stride=2, padding=1), 
                  nn.ReLU(inplace=True)]
        C_in = C_out
        C_mid, C_out = C//8, C//4
        H = floor((H-1)/2+1)
        extra.append(nn.Sequential(*layers))

    layers = [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
              nn.Conv2d(C_mid, C_out, 3), nn.ReLU(inplace=True)]
    extra.append(nn.Sequential(*layers))

    if H == 5:
        layers = [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                  nn.Conv2d(C_mid, C_out, 3), nn.ReLU(inplace=True)]
    elif H == 4:
        layers = [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                  nn.Conv2d(C_mid, C_out, 2), nn.ReLU(inplace=True)]
    extra.append(nn.Sequential(*layers))

    return extra

def VGG16_backbone(pretrained=False):
    """VGG-16 network, truncated at the conv5_3 layer, and split into 
    multiple Sequential segments to get sources from intermediate layers"""
    base = vgg16(pretrained=pretrained)
    base.features[16].ceil_mode = True # third MaxPool needs to be ceil_mode to give dimensions from SSD paper for 300x300 input
    backbone = [base.features[:23]] # source output at conv4_3

    """Final MaxPool modified, fc6/7 converted to conv (see SSD paper)"""
    layers = list(base.features[23:-1])
    layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    layers.append(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
    layers.append(nn.ReLU(inplace=True))
    backbone.append(nn.Sequential(*layers))
    return backbone

def default_boxes(image_size, feature_map_size, source_num, max_source, 
                  num_boxes):
    step_size = ceil(image_size/feature_map_size)
    scale = MIN_SCALE+source_num*(MAX_SCALE-MIN_SCALE)/(max_source-1)
    large_scale = MIN_SCALE+(source_num+1)*(MAX_SCALE-MIN_SCALE)/(max_source-1)
    boxes = []
    for idx, jdx in product(range(feature_map_size), repeat=2):
        center_x = (jdx+0.5)*step_size/image_size
        center_y = (idx+0.5)*step_size/image_size
        boxes += [center_x, center_y, scale, scale]
        boxes += [center_x, center_y, np.sqrt(scale*large_scale), np.sqrt(scale*large_scale)]
        for ratio in ASPECTS[:num_boxes-2]:
           boxes += [center_x, center_y, scale*np.sqrt(ratio), scale/np.sqrt(ratio)]
    return boxes

#def fc_to_conv(fc, C_in, C_out, )

