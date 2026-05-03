import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def adapt_first_conv_for_multichannel(conv_layer, in_channels):
    """
    Adapt pretrained EfficientNet-B0 first conv layer from 3 input channels
    to the number of channels in our tif input.

    Original EfficientNet-B0 first conv:
        Conv2d(3, 32, ...)

    Our tif input:
        Conv2d(in_channels, 32, ...)
    """
    old_weight = conv_layer.weight.data

    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    bias = conv_layer.bias is not None

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    with torch.no_grad():
        if in_channels == 3:
            new_conv.weight.copy_(old_weight)

        elif in_channels > 3:
            # Average RGB pretrained weights and repeat for all tif channels.
            # Scale by 3 / in_channels to keep activation magnitude stable.
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            new_weight = mean_weight.repeat(1, in_channels, 1, 1)
            new_weight = new_weight * (3.0 / in_channels)
            new_conv.weight.copy_(new_weight)

        else:
            new_conv.weight.copy_(old_weight[:, :in_channels, :, :])

    return new_conv


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=3, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.model = efficientnet_b0(weights=weights)

        # Change first convolution layer to accept tif input channels.
        self.model.features[0][0] = adapt_first_conv_for_multichannel(
            self.model.features[0][0],
            in_channels=in_channels,
        )

        # Change final classifier from ImageNet 1000 classes to our 3 clusters.
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)