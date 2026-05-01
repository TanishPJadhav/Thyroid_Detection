import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from .utils import get_device
from .data import preprocess_image


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        encoder = resnet34(weights=None)
        if in_channels != 3:
            old_conv = encoder.conv1
            encoder.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False)
        self.input_stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.center = nn.Sequential(ConvBNReLU(512, 512), ConvBNReLU(512, 512))
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.Sequential(ConvBNReLU(64, 32), nn.Conv2d(32, out_channels, kernel_size=1))

    def forward(self, x):
        x0 = self.input_stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        center = self.center(x4)
        d4 = self.dec4(center, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final_up(out)


def build_nodule_segmentation_model(backbone: str = 'resnet34', in_channels: int = 3, out_channels: int = 1):
    return ResUNet(in_channels=in_channels, out_channels=out_channels)


def predict_nodule_mask(model, image, device: str = 'cuda'):
    device = get_device(device)
    model = model.to(device)
    model.eval()
    img_t = preprocess_image(image, target_size=(512, 512)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = (torch.sigmoid(model(img_t)) > 0.5).float()
    return pred.squeeze(0).cpu()