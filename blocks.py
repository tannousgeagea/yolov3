import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    """
    This class implements a convolutional block that includes a Convolutional layer followed by Batch Normalization and SiLU activation.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Zero-padding added to both sides of the input
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.LeakyReLU(0.1)
        self.bn_act = bn_act

    def forward(self, x):
        return self.silu(self.bn(self.conv(x))) if self.bn_act else self.conv(x) if self.bn_act else self.conv(x)

class Bottleneck(nn.Module):
    """
    This class represents a single bottleneck block with a residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        shortcut (bool): If True, use the residual shortcut.
        e (float): Expansion factor for the bottleneck.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * e)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the bottleneck block.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_add:
            y += x
        return y

class ScalePrediction(nn.Module):
    """
    Defining a scale prediction module is essential for YOLOv3 as it processes feature maps at different scales 
    to detect objects of various sizes. The module usually consists of a few convolutional layers and outputs predictions 
    for bounding boxes, objectness scores, and class probabilities for each scale.

    Args:
        in_channels (int): The number of input channels (depth) of the feature map.
        num_classes (int): The number of object classes.
    """
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.attribute = num_classes +  5

        self.pred = nn.Sequential(
            ConvBNSiLU(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1),
            ConvBNSiLU(2 * in_channels, 3 * self.attribute, kernel_size=1, stride=1, padding=0, bn_act=False)
        )
    
    def forward(self, x):
        """
        Forward pass of the ScalePredicttion block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the Scale block.
        """
        return (
            self.pred(x).reshape(x.shape[0], 3, self.attribute, x.shape[2], x.shape[3]) # # Reshape to [batch, anchors, (num_classes + 4 + 1), height, width]
            .permute(0, 1, 3, 4, 2) # (N, 3, H. W, C + 4 + 1)
        )
    
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1, source=[-1]):
        super().__init__()
        self.d = dimension
        self.source = source

    def forward(self, x):
        return torch.cat(x, self.d)