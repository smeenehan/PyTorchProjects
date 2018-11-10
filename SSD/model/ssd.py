from math import floor
import torch
import torch.nn as nn
from

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
    backbone : torch.nn.Module
        Set of torch Module layers representing the backbone. This is assumed to
        be iterable (e.g., Sequential, ModuleList, etc.).
    bb_sources : tuple of ints
        Indices of layers within the backbone network to be used as sources.
    num_classes : int
        Number of output classes (should include a background class).
    """
    def __init__(self, input_shape, backbone, bb_sources, num_classes):
        self.num_classes = num_classes
        self.backbone = backbone
        self.bb_sources = bb_sources
        backbone_shape = self._backbone_shape(input_shape)
        self.extra, self.extra_sources = build_extra(backbone_shape)


    def forward(self, x):
        sources = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self.bb_sources:
                sources.append(x)

        for idx, layer in enumerate(self.extra):
            x = layer(x)
            if idx in self.extra_sources:
                sources.append(x)

        locs, classes = [], []
        for (x, l, c) in zip(sources, self.loc_layers, self.class_layers):
            locs.append(l(x).permute(0, 2, 3, 1))
            classes.append(c(x).permute(0, 2, 3, 1))

        locs = torch.cat([x.view(x.size(0), -1) for x in locs], 1)
        classes = torch.cat([x.view(x.size(0), -1) for x in classes], 1)
        output = (locs.view(locs.size(0), -1, 4),
                  classes.view(classes.size(0), -1, self.num_classes))

    def _backbone_shape(self, input_shape):
        """Determine output CHW of backbone network"""
        self.backbone.eval()
        with torch.no_grad():
            tens = torch.zeros(input_shape)[None, :]
            out = self.backbone(tens)
        return out.size()[1:]

    def _make_head(self):
        self.loc_layers = nn.ModuleList()
        self.class_layers = nn.ModuleList()
        num_bb_boxes = [6]*len(self.bb_sources)
        num_bb_boxes[0] = 4
        num_extra_boxes = [6]*len(self.extra_sources)
        num_extra_boxes[-2:] = [4, 4]

        for idx in self.bb_sources:
            C_out = self.backbone[idx].out_channels
            loc_layers.append(nn.Conv2d(C_out, num_bb_boxes[idx]*4, 3, padding=1))
            class_layers.append(nn.Conv2d(C_out, num_bb_boxes[idx]*self.num_classes, 
                                          3, padding=1))

        for idx in self.extra_sources:
            C_out = self.extra[idx].out_channels
            loc_layers.append(nn.Conv2d(C_out, num_extra_boxes[idx]*4, 3, padding=1))
            class_layers.append(nn.Conv2d(C_out, num_extra_boxes[idx]*self.num_classes, 
                                          3, padding=1))


def build_extra(input_shape):
    C, H, W = input_shape
    if H != W:
        raise ValueError('Only square inputs supported.')
    C_in, C_mid, C_out = C, C//4, C//2

    layers = []
    while H > 5:
        layers += [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                   nn.Conv2d(C_mid, C_out, 3, stride=2, padding=1), 
                   nn.ReLU(inplace=True)]
        C_in = C_out
        C_mid, C_out = C//8, C//4
        H = floor((H-1)/2+1)


    layers += [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
               nn.Conv2d(C_mid, C_out, 3), nn.ReLU(inplace=True)]

    if H == 5:
        layers += [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                   nn.Conv2d(C_mid, C_out, 3), nn.ReLU(inplace=True)]
    elif H == 4:
        layers += [nn.Conv2d(C_in, C_mid, 1), nn.ReLU(inplace=True),
                   nn.Conv2d(C_mid, C_out, 2), nn.ReLU(inplace=True)]

    sources = [3+4*x for x in range(len(layers)//4)]
    return nn.Sequential(*layers), sources

def VGG16_backbone():





