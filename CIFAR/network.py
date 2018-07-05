import torch
import torch.nn as nn

class ResNeXt(nn.Module):

    def __init__(self, input_shape, init_channels, blocks_per_stage, groups, 
                 num_classes):
        super().__init__()
        self.conv_init = nn.Conv2d(input_shape[0], init_channels, 3, bias=False)
        self.blocks = nn.ModuleList()
        input_channels = init_channels
        for stage_idx, num_blocks in enumerate(blocks_per_stage):
            mid_channels = init_channels*2**stage_idx
            output_channels = 4*mid_channels

            for block_idx in range(num_blocks):
                if block_idx==0 and stage_idx>0:
                    stride = 2
                else:
                    stride = 1

                self.blocks.append(ResNeXtBlock(
                    input_channels, mid_channels, output_channels, groups=groups,
                    stride=stride))
                input_channels = output_channels
        self.bn = nn.BatchNorm2d(output_channels)
        downsample = 2**(len(blocks_per_stage)-1)
        self.pool = nn.AvgPool2d((input_shape[1]/downsample, input_shape[2]/downsample))
        self.output_channels = output_channels
        self.fc = nn.Linear(output_channels, num_classes, bias=False)

    def forward(self, x):
        x = self.conv_init(x)
        for block in self.blocks:
            x = block(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view([-1, self.output_channels])
        x = self.fc(x)
        return x

class ResNeXtBlock(nn.Module):
    """Implement a residual block with aggregated transformations (i.e., grouped
    convolutions)

    Each block has a shortcut and residual path. The residual path performs a 1x1
    bottleneck convolution to an intermediate number of channels, multiple parallel
    grouped 3x3 convolutions followed by channel-wise concatenations, and a final
    un-bottleneck via 1x1 convolution. The residual path is then combined with the
    shortcut. BatchNorm and ReLu precede each convolution.

    The shortcut path is the identity by default, but if the input and output channel
    numbers differ and/or the residual path down-samples via striding, we compensate
    accordingly with a 1x1 convolution on the shortcut.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input.
    mid_channels : int
        Total number of channels in the intermediate residual convolutions.
    output_channels : int
        Number of channels in the output.
    groups : int
        Number of groups into which to separate the intermediate 3x3 residual 
        convolution. Defaults to 1.
    stride : int
        Stride used to downsample the input on the first convolution and shortcut
        path. Defaults to 1.
    """
    def __init__(self, input_channels, mid_channels, output_channels, 
                 groups=1, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, 1, bias=False, 
                               stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, bias=False,
                               padding=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, 1, bias=False)
        if input_channels!=output_channels or stride>1:
            self.conv_shortcut = nn.Conv2d(
                input_channels, output_channels, 1, bias=False, stride=stride)
        else:
            self.conv_shortcut = None

    def forward(self, x):
        x_in = self.bn1(x)
        x_in = nn.functional.relu(x_in)
        
        residual = self.conv1(x_in)
        residual = self.bn2(residual)
        residual = nn.functional.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn3(residual)
        residual = nn.functional.relu(residual)
        residual = self.conv3(residual)

        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(x_in)
        else:
            shortcut = x
        return shortcut+residual
