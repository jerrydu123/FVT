"""FVT-Tiny checkpoint-compatible model. SECOND uses 512; Landsat uses 384 internally."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from utils.misc import initialize_weights
from dinov3_my import EnhancedSimpleDino
from models.video_Tiny import BidirectionalResNet3D

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * out

class FAM(nn.Module):

    def __init__(self, c2=64, c3=128, c4=256, out=64):
        super().__init__()
        self.proj2 = nn.Conv2d(c2, out, 1, bias=False)
        self.proj3 = nn.Conv2d(c3, out, 1, bias=False)
        self.proj4 = nn.Conv2d(c4, out, 1, bias=False)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out * 3, out // 2, 1), nn.ReLU(inplace=True), nn.Conv2d(out // 2, out * 3, 1), nn.Sigmoid())
        self.channel_attention = ChannelAttention(out * 3)
        self.fuse = nn.Sequential(nn.Conv2d(out * 3, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True))

    def forward(self, x2, x3, x4):
        x3 = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        (p2, p3, p4) = (self.proj2(x2), self.proj3(x3), self.proj4(x4))
        cat = torch.cat([p2, p3, p4], dim=1)
        cat = self.se(cat) * cat
        cat = self.channel_attention(cat)
        return self.fuse(cat)

class Encoder(nn.Module):

    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        for (n, m) in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for (n, m) in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.fam = FAM(c2=128, c3=256, c4=512, out=128)
        self.conv = nn.Sequential(nn.Conv2d(512 + 256 + 128, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.conv)

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x4 = self.fam(x2, x3, x4)
        return (x0, x1, x4)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):

    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers_(in_channels, out_channels)
        self.cb = ConvBlock(out_channels)

    def _make_layers_(self, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        layers = [ResBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.cb(x)
        return x

class LCMDecoder(nn.Module):

    def __init__(self):
        super(LCMDecoder, self).__init__()
        self.db4_1 = DecoderBlock(128 + 64, 128)
        self.db1_0 = DecoderBlock(128 + 64, 128)

    def decode(self, x1, x2, db):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], 1)
        x = db(x)
        return x

    def forward(self, x0, x1, x4):
        x1 = self.decode(x4, x1, self.db4_1)
        x0 = self.decode(x1, x0, self.db1_0)
        return (x0, x1)

class CDBranch(nn.Module):

    def __init__(self):
        super(CDBranch, self).__init__()
        self.db4 = DecoderBlock(256 + 128, 128)
        self.db1 = DecoderBlock(256, 128)
        self.db0 = DecoderBlock(256, 128)

    def decode(self, x1, x2, db):
        x1 = db(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], 1)
        return x

    def forward(self, x0, x1, x1_4, x2_4):
        x4 = torch.cat([x1_4, x2_4, torch.abs(x1_4 - x2_4)], 1)
        x1 = self.decode(x4, x1, self.db4)
        x0 = self.decode(x1, x0, self.db1)
        x0 = self.db0(x0)
        return x0

class Fusion_DR(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Fusion_DR, self).__init__()
        self.conv = nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        fused = torch.cat([x1, x2], dim=1)
        out = self.conv(fused)
        out = self.bn(out)
        out = self.relu(out)
        return out

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

class FVT_Tiny(nn.Module):

    def __init__(self, channels=3, num_classes=7, dinov3_pretrained_path=None, resnet_pretrained=True, internal_size=512):
        super(FVT_Tiny, self).__init__()
        set_random_seed(42)
        if internal_size not in (384, 512):
            raise ValueError('Tiny internal_size must be 384 or 512')
        self.internal_size = internal_size
        self.video_len = 4
        self.num_query_tokens = 3
        self.encoder = Encoder(pretrained=resnet_pretrained)
        self.lcm_decoder = LCMDecoder()
        self.cd_branch = CDBranch()
        self.lcm_classifier1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, num_classes, kernel_size=1))
        self.lcm_classifier2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, num_classes, kernel_size=1))
        self.cd_classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
        self.conv1x1_a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv1x1_b = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.fusionDR0 = Fusion_DR(64, 64, 64)
        self.fusionDR1 = Fusion_DR(64, 64, 64)
        self.fusionDR4 = Fusion_DR(128, 64, 128)
        self.bidirectional_resnet3d4 = BidirectionalResNet3D(in_ch=128, hidden_ch=64, num_frames=10, use_light=True, interp_method='linear')
        self.dinounet = EnhancedSimpleDino(dinov3_pretrained_path=dinov3_pretrained_path, dinov3_model_name='dinounet_s', output_channels=64, use_cross_scale_fusion=True, pretrain_size=512)
        initialize_weights(self.lcm_decoder, self.cd_branch, self.lcm_classifier1, self.lcm_classifier2, self.cd_classifier)

    def forward(self, x1, x2, labels_bn=None, return_debug=False):
        x1_ori = x1
        x_size = x1_ori.size()
        x1 = F.interpolate(x1, size=(self.internal_size, self.internal_size), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(self.internal_size, self.internal_size), mode='bilinear', align_corners=False)
        (dx1_0, dx1_1, dx1_4) = self.dinounet(x1)
        (dx2_0, dx2_1, dx2_4) = self.dinounet(x2)
        (x1_0, x1_1, x1_4) = self.encoder(x1)
        (x2_0, x2_1, x2_4) = self.encoder(x2)
        x1_0_ori = x1_0
        x2_0_ori = x2_0
        x1_1_ori = x1_1
        x2_1_ori = x2_1
        x1_4_ori = x1_4
        x2_4_ori = x2_4
        x1_0 = self.fusionDR0(x1_0, dx1_0)
        x2_0 = self.fusionDR0(x2_0, dx2_0)
        x1_1 = self.fusionDR1(x1_1, dx1_1)
        x2_1 = self.fusionDR1(x2_1, dx2_1)
        x1_4 = self.fusionDR4(x1_4, dx1_4)
        x2_4 = self.fusionDR4(x2_4, dx2_4)
        x_size = x1_ori.size()
        if return_debug:
            (x1_4_enh, x2_4_enh, fused_2d, change_map, loss_bn, debug_3d) = self.bidirectional_resnet3d4(x1_4, x2_4, labels_bn, x_size, return_debug=True)
        else:
            (x1_4_enh, x2_4_enh, fused_2d, change_map, loss_bn) = self.bidirectional_resnet3d4(x1_4, x2_4, labels_bn, x_size, return_debug=False)
        (x1_0, x1_1) = self.lcm_decoder(x1_0, x1_1, x1_4_enh)
        (x2_0, x2_1) = self.lcm_decoder(x2_0, x2_1, x2_4_enh)
        cd_map = self.cd_branch(torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4_enh, x2_4_enh)
        change = self.cd_classifier(cd_map)
        out1 = self.lcm_classifier1(x1_0)
        out2 = self.lcm_classifier2(x2_0)
        if return_debug:
            debug_all = {'x1_4': x1_4, 'x2_4': x2_4, 'x1_4_enh': x1_4_enh, 'x2_4_enh': x2_4_enh, **debug_3d}
            return (F.interpolate(change, x_size[2:], mode='bilinear'), F.interpolate(out1, x_size[2:], mode='bilinear'), F.interpolate(out2, x_size[2:], mode='bilinear'), loss_bn, debug_all)
        return (F.interpolate(change, x_size[2:], mode='bilinear'), F.interpolate(out1, x_size[2:], mode='bilinear'), F.interpolate(out2, x_size[2:], mode='bilinear'), loss_bn)
