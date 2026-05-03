import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


def adapt_first_conv_for_multichannel(conv_layer, in_channels):
    """
    Adapt pretrained ResNet18 first conv layer from 3 input channels
    to the number of channels in our tif input.

    Original ResNet18 conv1:
        Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

    Our tif input:
        Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
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
            # Average pretrained RGB weights and repeat them for all tif channels.
            # Scale by 3 / in_channels to keep activation magnitude closer to original ResNet.
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            new_weight = mean_weight.repeat(1, in_channels, 1, 1)
            new_weight = new_weight * (3.0 / in_channels)
            new_conv.weight.copy_(new_weight)

        else:
            # If fewer than 3 channels, use the first in_channels pretrained channels.
            new_conv.weight.copy_(old_weight[:, :in_channels, :, :])

    return new_conv


class ResNet18Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=3, pretrained=True):
        super(ResNet18Classifier, self).__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.model = resnet18(weights=weights)

        # Change first conv layer from 3-channel RGB to tif input channels.
        self.model.conv1 = adapt_first_conv_for_multichannel(
            self.model.conv1,
            in_channels=in_channels,
        )

        # Change final classifier from ImageNet 1000 classes to 3 clusters.
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18Classifier(
        in_channels=8,
        num_classes=3,
        pretrained=False,
    )

    print(model)
    print("ResNet18Classifier model created successfully.")