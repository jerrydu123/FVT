import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Type, Union, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dinounet.dinov3.models.vision_transformer import DinoVisionTransformer
from dinounet.dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16


DINOv3_MODEL_INFO = {
    "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}

DINOv3_INTERACTION_INDEXES = {
    "dinounet_s": [2, 5, 8, 11],
    "dinounet_b": [2, 5, 8, 11],
    "dinounet_l": [4, 11, 17, 23],
    "dinounet_7b": [9, 19, 29, 39],
}

DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinounet_7b": dinov3_vit7b16,
}


def load_dinov3_model(model_name: str, pretrained_path: str = None):
    if model_name not in DINOv3_MODEL_FACTORIES:
        supported = list(DINOv3_MODEL_FACTORIES.keys())
        raise ValueError(f"Unsupported model: {model_name}. Supported: {supported}")

    model_factory = DINOv3_MODEL_FACTORIES[model_name]

    if pretrained_path is not None:
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"DINOv3 checkpoint does not exist: {pretrained_path}")
        print(f"🔄 Loading custom pretrained weights from {pretrained_path}")
        model = model_factory(pretrained=False)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        print(f"✅ Loaded custom weights")
    else:
        print(f"🔄 Initializing random weights for {model_name}")
        model = model_factory(pretrained=False)
        print(f"✅ Initialized random weights")

    return model


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """Apply channel attention followed by spatial attention."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class EnhancedMultiScaleConv(nn.Module):
    """Fuse depthwise-separable convolution branches with CBAM and a residual path."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()

        mid_ch = max(out_channels // 2, 32)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1, bias=bias),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

        self.branch1 = self._make_branch(mid_ch, mid_ch, 3, 1, bias)
        self.branch2 = self._make_branch(mid_ch, mid_ch, 5, 2, bias)
        self.branch3 = self._make_branch(mid_ch, mid_ch, 7, 3, bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 3, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(out_channels, reduction=8)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def _make_branch(self, in_ch, out_ch, k, p, bias):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, padding=p, groups=in_ch, bias=bias),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.reduce(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)
        out = self.cbam(out)

        return out + identity


class CrossScaleFusion(nn.Module):
    """Fuse coarse semantic features with finer spatial features."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv_high = nn.Conv2d(channels, channels, 1)
        self.conv_low = nn.Conv2d(channels, channels, 1)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.attention = CBAM(channels, reduction=8)

    def forward(self, high_feat, low_feat):
        h, w = low_feat.shape[2:]
        high_up = F.interpolate(high_feat, size=(h, w), mode='bilinear', align_corners=False)
        high_up = self.conv_high(high_up)
        low_feat = self.conv_low(low_feat)

        fused = torch.cat([high_up, low_feat], dim=1)
        fused = self.conv_fuse(fused)
        fused = self.attention(fused)

        return fused + low_feat


class EnhancedDINOv3Adapter(nn.Module):
    """Extract frozen DINOv3 features at selected layers and resize them to four scales."""
    def __init__(
        self,
        backbone,
        interaction_indexes: List[int] = [9, 19, 29, 39],
        pretrain_size: int = 512,
        use_cls_token: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.interaction_indexes = interaction_indexes
        self.pretrain_size = pretrain_size
        self.patch_size = backbone.patch_size
        self.embed_dim = backbone.embed_dim
        self.use_cls_token = use_cls_token

        print(f"✅ Enhanced DINOv3 Adapter initialized")
        print(f"   Embed dim: {self.embed_dim}")
        print(f"   Patch size: {self.patch_size}")
        print(f"   Interaction indexes: {self.interaction_indexes}")

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        H_toks = H // self.patch_size
        W_toks = W // self.patch_size

        all_layers = self.backbone.get_intermediate_layers(
            x, n=self.interaction_indexes, return_class_token=self.use_cls_token
        )

        features = []
        for layer_output in all_layers:
            if self.use_cls_token and isinstance(layer_output, tuple):
                feat = layer_output[0]
            else:
                feat = layer_output
            feat = feat.transpose(1, 2).view(B, self.embed_dim, H_toks, W_toks).contiguous()
            features.append(feat)

        f1 = F.interpolate(features[0], size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        f2 = F.interpolate(features[1], size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        f3 = F.interpolate(features[2], size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        f4 = F.interpolate(features[3], size=(H // 16, W // 16), mode='bilinear', align_corners=False)

        return {"1": f1, "2": f2, "3": f3, "4": f4, "raw": features}


class EnhancedDINOv3EncoderAdapter(nn.Module):
    """Project DINOv3 features with multiscale convolutions and optional cross-scale fusion."""
    def __init__(
        self,
        dinov3_adapter: EnhancedDINOv3Adapter,
        target_channels: List[int],
        use_cross_scale_fusion: bool = True,
        conv_bias: bool = False
    ):
        super().__init__()

        self.dinov3_adapter = dinov3_adapter
        self.target_channels = target_channels
        self.use_cross_scale_fusion = use_cross_scale_fusion

        embed_dim = self.dinov3_adapter.embed_dim

        self.projections = nn.ModuleList([
            EnhancedMultiScaleConv(embed_dim, ch, bias=conv_bias)
            for ch in target_channels
        ])

        if use_cross_scale_fusion:
            self.cross_fusions = nn.ModuleList([
                CrossScaleFusion(target_channels[i])
                for i in range(len(target_channels) - 1)
            ])

        self.output_channels = target_channels
        self.strides = [[2, 2]] * len(target_channels)
        self.kernel_sizes = [[3, 3]] * len(target_channels)

        print(f"✅ Enhanced DINOv3EncoderAdapter initialized")
        print(f"   Input dim: {embed_dim}")
        print(f"   Target channels: {target_channels}")
        print(f"   Cross-scale fusion: {use_cross_scale_fusion}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape

        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            x = x[:, :3, :, :] if C > 3 else x.repeat(1, 3//C + 1, 1, 1)[:, :3, :, :]

        features_dict = self.dinov3_adapter(x)

        feature_keys = ["1", "2", "3", "4"]
        target_scales = [2, 4, 8, 16]

        projected = []
        for i, key in enumerate(feature_keys):
            feat = features_dict[key]
            proj = self.projections[i](feat)

            tgt_h, tgt_w = H // target_scales[i], W // target_scales[i]
            if proj.shape[2] != tgt_h or proj.shape[3] != tgt_w:
                proj = F.interpolate(proj, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)

            projected.append(proj)

        # Propagate coarse semantic features to finer scales.
        if self.use_cross_scale_fusion:
            for i in range(len(projected) - 2, -1, -1):
                projected[i] = self.cross_fusions[i](projected[i + 1], projected[i])

        return projected

    def compute_conv_feature_map_size(self, input_size):
        return 0


class EnhancedSimpleDino(nn.Module):
    """Expose three DINOv3 feature scales for fusion with the ResNet encoder."""
    def __init__(
        self,
        dinov3_pretrained_path: str,
        dinov3_model_name: str = "dinounet_l",
        output_channels: int = 64,
        use_cross_scale_fusion: bool = True,
        conv_bias: bool = False,
        pretrain_size: int = 512
    ):
        super().__init__()

        if dinov3_model_name not in DINOv3_MODEL_INFO:
            raise ValueError(f"Unknown model: {dinov3_model_name}")

        model_info = DINOv3_MODEL_INFO[dinov3_model_name]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[dinov3_model_name]

        print(f"\n{'='*70}")
        print(f"🔧 Creating Enhanced SimpleDino")
        print(f"{'='*70}")
        print(f"Model: {dinov3_model_name}")
        print(f"Embedding dim: {model_info['embed_dim']}")
        print(f"Output channels: {output_channels}")
        print(f"Cross-scale fusion: {use_cross_scale_fusion}")
        print(f"Improvements: CBAM, DepthwiseSeparable, CrossScaleFusion, Residual")
        print(f"{'='*70}\n")

        dinov3_backbone = load_dinov3_model(dinov3_model_name, dinov3_pretrained_path)

        dinov3_adapter = EnhancedDINOv3Adapter(
            backbone=dinov3_backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=pretrain_size
        )

        target_channels = [output_channels] * 4

        self.encoder = EnhancedDINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=target_channels,
            use_cross_scale_fusion=use_cross_scale_fusion,
            conv_bias=conv_bias
        )

        print("✅ Enhanced SimpleDino initialized successfully")
        print(f"📊 Output scales: 1/4, 1/8, 1/16 @ {output_channels} channels\n")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return feature maps at 1/2, 1/4, and 1/8 of the input resolution."""
        skips = self.encoder(x)
        return skips[0], skips[1], skips[2]
