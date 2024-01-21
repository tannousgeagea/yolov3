import torch
import logging
import torch.nn as nn
from blocks import Concat
from blocks import ConvBNSiLU
from blocks  import Bottleneck
from blocks import ScalePrediction

architecture = [
    # [from, number, module, args=(kernel_size, filter, stride, padding)]
    [-1, 1, ConvBNSiLU, [3, 32, 1, 1]],
    [-1, 1, ConvBNSiLU, [3, 64, 2, 1]],
    [-1, 1, Bottleneck, [64, True]],
    [-1, 1, ConvBNSiLU, [3, 128, 2, 1]],
    [-1, 2, Bottleneck, [128, True]],
    [-1, 1, ConvBNSiLU, [3, 256, 2, 1]],
    [-1, 8, Bottleneck, [256, True]],
    [-1, 1, ConvBNSiLU, [3, 512, 2, 1]],
    [-1, 8, Bottleneck, [512, True]],
    [-1, 1, ConvBNSiLU, [3, 1024, 2, 1]],
    [-1, 4, Bottleneck, [1024, True]],
    
    # first scale
    [-1, 1, ConvBNSiLU, [1, 512, 1, 0]],
    [-1, 1, ConvBNSiLU, [3, 1024, 1, 1]],
    [-1, 1, Bottleneck, [1024, False]],
    [-1, 1, ConvBNSiLU, [3, 512, 1, 1]],
    [-1, 1, ScalePrediction, 512], 

    # 2nd scale
    [-1, 1, ConvBNSiLU, [1, 256, 1, 0]],
    [-1, 1, nn.Upsample, [2, "nearest"]],
    [[-1, 8], 1, Concat, [1]],
    [-1, 1, ConvBNSiLU, [1, 256, 1, 0]],
    [-1, 1, ConvBNSiLU, [3, 512, 1, 1]],
    [-1, 1, Bottleneck, [512, False]],
    [-1, 1, ConvBNSiLU, [1, 256, 1, 0]],
    [-1, 1, ScalePrediction, 256],

    # 3rd scale
    [-1, 1, ConvBNSiLU, [1, 128, 1, 0]],
    [-1, 1, nn.Upsample, [2, 'nearest']],
    [[-1, 6], 1, Concat, [1]],
    [-1, 1, ConvBNSiLU, [1, 128, 1, 0]],
    [-1, 1, ConvBNSiLU, [3, 256, 1, 1]],
    [-1, 1, Bottleneck, [256, False]],
    [-1, 1, ConvBNSiLU, [1, 128, 1, 0]],
    [-1, 1, ScalePrediction, 128],
]



class Yolov3(nn.Module):
    """
    This class represents as implementation of Yolov3
    
    Args:
        in_channels (int): Number of channels in the input image
        **kwargs: set of additional arguments
    """
    def __init__(self, in_channels, num_classes=80, **kwargs):
        super(Yolov3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self.parse_module(architecture)

    def forward(self, x):
        outputs = []
        routes = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            if isinstance(layer, Concat):
                x = layer((x, routes[layer.source[1]]))
                continue
            
            x = layer(x)

            routes.append(x)
            

        return outputs

    def _make_layer(self, n=1, channels=64):
        return nn.Sequential(*(Bottleneck(channels, channels, shortcut=True, e=0.5) for _ in range(n)))
    
    def parse_module(self, configurations):
        layers = []
        in_channels = self.in_channels
        try:
            for i, layer in enumerate(configurations):
                f, rep, module, args = layer
                if module is ConvBNSiLU:
                    kernel_size, out_channels, stride, padding = args
                    layers += [module(
                        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
                        )
                    ]
                    
                    in_channels = out_channels
                
                elif module is Bottleneck:
                    in_channels, shortcut = args
                    layers += [nn.Sequential(*(module(in_channels, in_channels, shortcut=shortcut, e=0.5) for _ in range(rep)))]

                elif module is ScalePrediction:
                    layers += [
                        module(in_channels=args, num_classes=self.num_classes)
                    ]
                
                elif module is Concat:
                    layers += [
                        module(dimension=args[0], source=f)
                    ]
                    in_channels = configurations[f[1]][3][0] + in_channels
    
                elif module is nn.Upsample:
                    layers += [
                        module(scale_factor=args[0], mode=args[1])
                    ]

        except Exception as err:
            logging.error('Unexpected Error while parsing Conv layers: %s' %err)    

        return layers


if __name__ == "__main__":
    x = torch.randn((2, 3, 416, 416))
    model = Yolov3(in_channels=3, num_classes=80)
    out = model(x)
    print(f"Scale 1: {out[0].shape}")
    print(f"Scale 2: {out[1].shape}")
    print(f"Scale 3: {out[2].shape}")
