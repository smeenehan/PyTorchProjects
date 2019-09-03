from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

BOX_COLORS = ['red', 'blue', 'green', 'cyan', 'magenta']

def show_detections(image, targets, class_names):
    _, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)
    add_boxes_and_labels(ax, targets, class_names)
    plt.show()

def add_boxes_and_labels(axes, targets, class_names, colors=None):
    if colors is None:
        colors = BOX_COLORS
    text_bbox = {'alpha': 0.5, 'pad': 2, 'edgecolor': 'none'}
    for target, color in zip(targets, cycle(colors)):
        """Use 1-y2 as the box origin (lower left) due to the way that imshow works, 
        with image pixel (0, 0) in the upper left"""
        x, y, w, h, id_num = target[1], 1-target[2], target[3]-target[1], \
                             target[2]-target[0], int(target[4])
        patch = patches.Rectangle((x, y), w, h, transform=axes.transAxes,
                                  linewidth=2, edgecolor=color, facecolor='none')
        axes.add_patch(patch)

        caption = class_names[id_num]
        text_bbox['facecolor'] = color
        axes.text(x, y, caption, size=12, color='white', backgroundcolor='none',
                  verticalalignment='top', transform=axes.transAxes, bbox=text_bbox)