import torch
import torch.nn as nn
import torch.nn.functional as F
from models.video_323 import SequenceGeneratorFactory


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, temporal_stride=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=(temporal_stride, stride, stride),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
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


class LightResBlock3D(nn.Module):
    """Factorize 3D convolutions into temporal and spatial operations."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, temporal_stride=1):
        super(LightResBlock3D, self).__init__()

        self.temporal_conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(3, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn_t1 = nn.BatchNorm3d(out_channels)

        self.spatial_conv1 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False
        )
        self.bn_s1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.temporal_conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn_t2 = nn.BatchNorm3d(out_channels)

        self.spatial_conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            bias=False
        )
        self.bn_s2 = nn.BatchNorm3d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.temporal_conv1(x)
        out = self.bn_t1(out)
        out = self.relu(out)
        out = self.spatial_conv1(out)
        out = self.bn_s1(out)
        out = self.relu(out)
        out = self.temporal_conv2(out)
        out = self.bn_t2(out)
        out = self.relu(out)
        out = self.spatial_conv2(out)
        out = self.bn_s2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MultiScaleFrameDiff(nn.Module):
    """Fuse frame differences at gaps of one, two, and the full sequence."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.diff_1 = self._make_diff_layer(in_channels, out_channels, gap=1)
        self.diff_2 = self._make_diff_layer(in_channels, out_channels, gap=2)
        self.diff_all = self._make_diff_layer(in_channels, out_channels, gap=-1)

        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_diff_layer(self, in_ch, out_ch, gap):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                     padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def compute_gap_diff(self, x, gap):
        if gap == -1:
            diff = x[:, :, -1:] - x[:, :, :1]
            diff = diff.repeat(1, 1, x.size(2), 1, 1)
        else:
            diff = x[:, :, gap:] - x[:, :, :-gap]
            diff = F.pad(diff, (0, 0, 0, 0, gap, 0), mode='replicate')
        return diff

    def forward(self, x):
        diff_1 = self.compute_gap_diff(x, 1)
        diff_2 = self.compute_gap_diff(x, 2)
        diff_all = self.compute_gap_diff(x, -1)
        feat_1 = self.diff_1(diff_1)
        feat_2 = self.diff_2(diff_2)
        feat_all = self.diff_all(diff_all)
        fused = self.fusion(torch.cat([feat_1, feat_2, feat_all], dim=1))
        return fused


class DiffAwareTemporalAggregator(nn.Module):
    """Fuse attention-weighted and mean temporal features and estimate change intensity."""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.change_estimator = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.temporal_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_ch, in_ch // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch // 4, in_ch, kernel_size=1),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        change_intensity = self.change_estimator(x)
        attn = self.temporal_attn(x)
        attn = torch.softmax(attn, dim=2)
        weighted = (x * attn).sum(dim=2)
        avg = x.mean(dim=2)
        aggregated = self.fusion(torch.cat([weighted, avg], dim=1))
        change_map = change_intensity.max(dim=2)[0]
        return aggregated, change_map


class BidirectionalResNet3D(nn.Module):
    """Model bidirectional change with spatial downsampling and reduced hidden channels."""
    def __init__(self, in_ch=128, hidden_ch=64, num_frames=4, use_light=True, interp_method='tanh'):
        super().__init__()
        self.num_frames = num_frames
        self.use_light = use_light
        self.interp_method = interp_method
        self.hidden_ch = hidden_ch

        if interp_method != 'linear':
            self.seq_generator = SequenceGeneratorFactory.build(
                interp_method,
                in_ch=in_ch,
                num_frames=num_frames
            )
        else:
            self.seq_generator = None

        self.input_compress = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True)
        )

        self.output_project = nn.Sequential(
            nn.Conv2d(hidden_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        ResBlock = LightResBlock3D if use_light else ResBlock3D

        self.forward_layer1 = self._make_layer(ResBlock, hidden_ch, hidden_ch, blocks=2)
        self.forward_layer2 = self._make_layer(ResBlock, hidden_ch, hidden_ch, blocks=2)
        self.backward_layer1 = self._make_layer(ResBlock, hidden_ch, hidden_ch, blocks=2)
        self.backward_layer2 = self._make_layer(ResBlock, hidden_ch, hidden_ch, blocks=2)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.diff_enhancer_forward = MultiScaleFrameDiff(hidden_ch, hidden_ch)
        self.diff_enhancer_backward = MultiScaleFrameDiff(hidden_ch, hidden_ch)

        downsample_3d = nn.Sequential(
            nn.Conv3d(hidden_ch * 2, hidden_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(hidden_ch)
        )

        self.bidirectional_fusion = nn.Sequential(
            ResBlock(hidden_ch * 2, hidden_ch, stride=1, downsample=downsample_3d),
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_ch),
            nn.ReLU(inplace=True)
        )

        self.temporal_aggregator = DiffAwareTemporalAggregator(hidden_ch, hidden_ch)
        self.conv1x1_loss = nn.Conv2d(hidden_ch, out_channels=1, kernel_size=1)

    def _make_layer(self, block, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def generate_sequence(self, start, end, num_frames, return_alphas=False):
        if self.interp_method == 'linear' or self.seq_generator is None:
            sequence = []
            alphas = []
            for i in range(num_frames):
                alpha = i / (num_frames - 1)
                alphas.append(alpha)
                sequence.append(start * (1 - alpha) + end * alpha)
            sequence = torch.stack(sequence, dim=2)
            if return_alphas:
                alphas = torch.tensor(alphas, device=start.device, dtype=start.dtype)
                return sequence, alphas
            return sequence
        else:
            sequence = self.seq_generator(start, end)
            if return_alphas:
                alphas = torch.linspace(0, 1, steps=num_frames, device=start.device, dtype=start.dtype)
                return sequence, alphas
            return sequence

    def forward(self, x1, x2, labels_bn, x_size, return_debug=False):
        x1_compressed = self.input_compress(x1)
        x2_compressed = self.input_compress(x2)
        orig_h, orig_w = x1_compressed.shape[2], x1_compressed.shape[3]

        forward_seq, alphas = self.generate_sequence(x1_compressed, x2_compressed, self.num_frames, return_alphas=True)
        backward_seq, _ = self.generate_sequence(x2_compressed, x1_compressed, self.num_frames, return_alphas=True)

        if return_debug:
            forward_seq.retain_grad()
            backward_seq.retain_grad()

        forward_feat = self.forward_layer1(forward_seq)
        forward_feat = self.forward_layer2(forward_feat)
        forward_feat_enhanced = forward_feat + self.diff_enhancer_forward(forward_feat)

        backward_feat = self.backward_layer1(backward_seq)
        # Preserve the historical checkpoint computation: layer1 output is discarded.
        backward_feat = self.backward_layer2(backward_seq)
        backward_feat_enhanced = backward_feat + self.diff_enhancer_backward(backward_feat)

        bi_feat = torch.cat([forward_feat_enhanced, backward_feat_enhanced], dim=1)
        fused_3d = self.bidirectional_fusion(bi_feat)

        fused_2d, change_map = self.temporal_aggregator(fused_3d)

        fused_2d = F.interpolate(fused_2d, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        feature_upsampled = self.conv1x1_loss(fused_2d)

        if isinstance(x_size, (tuple, list)) and len(x_size) >= 2:
            target_hw = x_size[-2:]
        else:
            target_hw = feature_upsampled.shape[-2:]

        feature_upsampled = F.interpolate(feature_upsampled, size=target_hw, mode='bilinear', align_corners=False)

        loss_bn = None
        if labels_bn is not None:
            if labels_bn.dim() == 5 and labels_bn.size(2) == 1:
                labels_bn = labels_bn.squeeze(2)
            loss_bn = self.bce_loss(feature_upsampled, labels_bn)

        # Restore channels and spatial resolution before adding the residual.
        fused_2d_proj = self.output_project(fused_2d)
        enhanced_x1 = x1 + F.interpolate(fused_2d_proj, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)
        enhanced_x2 = x2 + F.interpolate(fused_2d_proj, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)

        if return_debug:
            debug = {
                "alphas": alphas,
                "forward_seq": forward_seq,
                "backward_seq": backward_seq,
                "forward_feat": forward_feat,
                "backward_feat": backward_feat,
                "forward_feat_enhanced": forward_feat_enhanced,
                "backward_feat_enhanced": backward_feat_enhanced,
                "fused_3d": fused_3d,
                "fused_2d": fused_2d,
                "change_map": change_map,
            }
            return enhanced_x1, enhanced_x2, fused_2d, change_map, loss_bn, debug

        return enhanced_x1, enhanced_x2, fused_2d, change_map, loss_bn
