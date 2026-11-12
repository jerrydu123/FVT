import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            frames.append(x1 * (1 - t) + x2 * t)
        return torch.stack(frames, dim=2)


class QuadraticSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4, mode='ease_in'):
        super().__init__()
        self.num_frames = num_frames
        self.mode = mode

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            if self.mode == 'ease_in':
                alpha = t ** 2
            elif self.mode == 'ease_out':
                alpha = 1 - (1 - t) ** 2
            else:
                alpha = t
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class CubicSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = 3 * t**2 - 2 * t**3
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class SmootherStepSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = 6 * t**5 - 15 * t**4 + 10 * t**3
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class GammaSequenceGenerator(nn.Module):
    """Use gamma > 1 for ease-in, gamma < 1 for ease-out, and gamma = 1 for linear interpolation."""
    def __init__(self, num_frames=4, gamma=2.0):
        super().__init__()
        self.num_frames = num_frames
        self.gamma = gamma

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = t ** self.gamma
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class SineEaseInSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = 1.0 - math.cos((math.pi * t) / 2.0)
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class SineEaseOutSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = math.sin((math.pi * t) / 2.0)
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class CosineEaseInOutSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class ExponentialEaseInSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4, base=2.0):
        super().__init__()
        self.num_frames = num_frames
        self.base = base

    def forward(self, x1, x2):
        frames = []
        denom = self.base - 1.0 + 1e-8
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = (self.base ** t - 1.0) / denom
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class ExponentialEaseOutSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4, base=2.0):
        super().__init__()
        self.num_frames = num_frames
        self.base = base

    def forward(self, x1, x2):
        frames = []
        denom = self.base - 1.0 + 1e-8
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            alpha = 1.0 - (self.base ** (1.0 - t) - 1.0) / denom
            frames.append(x1 * (1 - alpha) + x2 * alpha)
        return torch.stack(frames, dim=2)


class SigmoidSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4, sharpness=10.0):
        super().__init__()
        self.num_frames = num_frames
        self.sharpness = sharpness

    def forward(self, x1, x2):
        frames = []

        a0 = torch.sigmoid(torch.tensor(-0.5 * self.sharpness, dtype=x1.dtype, device=x1.device))
        a1 = torch.sigmoid(torch.tensor( 0.5 * self.sharpness, dtype=x1.dtype, device=x1.device))

        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            tt = torch.tensor(t, dtype=x1.dtype, device=x1.device)
            alpha = torch.sigmoid(self.sharpness * (tt - 0.5))
            alpha = (alpha - a0) / (a1 - a0 + 1e-8)
            frames.append(x1 * (1 - alpha) + x2 * alpha)

        return torch.stack(frames, dim=2)


class TanhSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4, sharpness=3.0):
        super().__init__()
        self.num_frames = num_frames
        self.sharpness = sharpness

    def forward(self, x1, x2):
        frames = []

        a0 = 0.5 * (1.0 + torch.tanh(torch.tensor(-0.5 * self.sharpness, dtype=x1.dtype, device=x1.device)))
        a1 = 0.5 * (1.0 + torch.tanh(torch.tensor( 0.5 * self.sharpness, dtype=x1.dtype, device=x1.device)))

        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            tt = torch.tensor(t, dtype=x1.dtype, device=x1.device)
            alpha = 0.5 * (1.0 + torch.tanh(self.sharpness * (tt - 0.5)))
            alpha = (alpha - a0) / (a1 - a0 + 1e-8)
            frames.append(x1 * (1 - alpha) + x2 * alpha)

        return torch.stack(frames, dim=2)


class SlerpSequenceGenerator(nn.Module):
    def __init__(self, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

    def slerp(self, v0, v1, t):
        """Interpolate [B, C, H, W] feature vectors over t in [0, 1]."""
        B, C, H, W = v0.shape

        v0_f = v0.flatten(2)
        v1_f = v1.flatten(2)

        v0_n = F.normalize(v0_f, dim=1)
        v1_n = F.normalize(v1_f, dim=1)

        dot = (v0_n * v1_n).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta).clamp(min=1e-6)

        coeff0 = torch.sin((1.0 - t) * theta) / sin_theta
        coeff1 = torch.sin(t * theta) / sin_theta

        # Fall back to linear interpolation at small angles for numerical stability.
        fallback = theta < 1e-4
        coeff0 = torch.where(fallback, torch.full_like(coeff0, 1.0 - t), coeff0)
        coeff1 = torch.where(fallback, torch.full_like(coeff1, t), coeff1)

        interp = coeff0 * v0_f + coeff1 * v1_f
        return interp.view(B, C, H, W)

    def forward(self, x1, x2):
        frames = []
        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            frames.append(self.slerp(x1, x2, t))
        return torch.stack(frames, dim=2)


class LearnableInterpolator(nn.Module):
    """Learn a residual over linear interpolation using a 1x1 convolutional MLP."""
    def __init__(self, in_ch, hidden_ch=64, num_frames=4, time_dim=8):
        super().__init__()
        self.num_frames = num_frames
        self.time_dim = time_dim

        self.interpolator = nn.Sequential(
            nn.Conv2d(in_ch * 2 + time_dim, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=1, bias=False),
        )

    def _fourier_time_embed(self, t, dim=8):
        freqs = torch.arange(dim // 2, device=t.device, dtype=t.dtype)
        freqs = torch.pow(torch.tensor(10.0, device=t.device, dtype=t.dtype), -freqs / max(1, (dim // 2)))
        embed = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)
        return embed

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        frames = []

        for i in range(self.num_frames):
            t = torch.tensor(i / (self.num_frames - 1), dtype=x1.dtype, device=x1.device)
            t_feat = self._fourier_time_embed(t, dim=self.time_dim)
            t_feat = t_feat.view(1, self.time_dim, 1, 1).expand(B, self.time_dim, H, W)

            inp = torch.cat([x1, x2, t_feat], dim=1)
            linear_base = x1 * (1 - t.item()) + x2 * t.item()
            delta = self.interpolator(inp)
            frames.append(linear_base + delta)

        return torch.stack(frames, dim=2)


class FeatureWarpSequenceGenerator(nn.Module):
    """Generate intermediate features by estimating flow and warping the first input."""
    def __init__(self, in_ch=128, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

        self.flow_estimator = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 2, kernel_size=3, padding=1),
        )

    def warp(self, feature, flow, t):
        B, C, H, W = feature.shape
        scaled_flow = flow * t

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1, 1, W, device=feature.device, dtype=feature.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, H, W, 2)

        flow_norm = scaled_flow.permute(0, 2, 3, 1).clone()
        flow_norm[..., 0] = flow_norm[..., 0] / ((W - 1) / 2 + 1e-6)
        flow_norm[..., 1] = flow_norm[..., 1] / ((H - 1) / 2 + 1e-6)

        sample_grid = base_grid + flow_norm
        warped = F.grid_sample(
            feature,
            sample_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        return warped

    def forward(self, x1, x2):
        flow = self.flow_estimator(torch.cat([x1, x2], dim=1))
        frames = []

        for i in range(self.num_frames):
            t = i / (self.num_frames - 1)
            warped = self.warp(x1, flow, t)
            alpha = t
            frame = (1 - alpha) * warped + alpha * x2
            frames.append(frame)

        return torch.stack(frames, dim=2)


class AttentionGuidedInterpolator(nn.Module):
    """Learn a spatially varying nonlinear time mapping guided by change attention."""
    def __init__(self, in_ch, num_frames=4):
        super().__init__()
        self.num_frames = num_frames

        self.change_attention = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.time_mapper = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 2, kernel_size=1),
            nn.Softplus()
        )

    def beta_cdf_approx(self, t, alpha, beta):
        t_a = t ** alpha.clamp(0.1, 10.0)
        one_minus_t_b = (1 - t + 1e-6) ** beta.clamp(0.1, 10.0)
        return t_a / (t_a + one_minus_t_b + 1e-6)

    def forward(self, x1, x2):
        pair = torch.cat([x1, x2], dim=1)

        change_map = self.change_attention(pair)
        time_params = self.time_mapper(pair)
        alpha = time_params[:, 0:1]
        beta = time_params[:, 1:2]

        frames = []
        for i in range(self.num_frames):
            t_linear = i / (self.num_frames - 1)
            t_tensor = torch.tensor(t_linear, dtype=x1.dtype, device=x1.device)

            t_nonlinear = self.beta_cdf_approx(t_tensor, alpha, beta)
            t_final = change_map * t_nonlinear + (1 - change_map) * t_linear

            frame = x1 * (1 - t_final) + x2 * t_final
            frames.append(frame)

        return torch.stack(frames, dim=2)


class SequenceGeneratorFactory:
    METHODS = {
        'linear':        LinearSequenceGenerator,
        'quadratic':     QuadraticSequenceGenerator,
        'quadratic_in':  QuadraticSequenceGenerator,
        'quadratic_out': QuadraticSequenceGenerator,
        'cubic':         CubicSequenceGenerator,
        'smootherstep':  SmootherStepSequenceGenerator,
        'gamma':         GammaSequenceGenerator,

        'sine_in':       SineEaseInSequenceGenerator,
        'sine_out':      SineEaseOutSequenceGenerator,
        'cosine':        CosineEaseInOutSequenceGenerator,

        'exp_in':        ExponentialEaseInSequenceGenerator,
        'exp_out':       ExponentialEaseOutSequenceGenerator,
        'sigmoid':       SigmoidSequenceGenerator,
        'tanh':          TanhSequenceGenerator,

        'slerp':         SlerpSequenceGenerator,
        'learnable':     LearnableInterpolator,
        'warp':          FeatureWarpSequenceGenerator,
        'attention':     AttentionGuidedInterpolator,
    }

    @classmethod
    def available_methods(cls):
        return list(cls.METHODS.keys())

    @classmethod
    def build(cls, method, in_ch=None, num_frames=4, **kwargs):
        if method not in cls.METHODS:
            raise ValueError(f"Unknown interpolation method: {method}. "
                             f"Available: {cls.available_methods()}")

        gen_cls = cls.METHODS[method]

        if method == 'quadratic_in':
            return gen_cls(num_frames=num_frames, mode='ease_in')
        if method == 'quadratic_out':
            return gen_cls(num_frames=num_frames, mode='ease_out')
        if method == 'quadratic':
            return gen_cls(num_frames=num_frames, mode=kwargs.pop('mode', 'ease_in'))

        if method in ['learnable', 'attention']:
            if in_ch is None:
                raise ValueError(f"Method '{method}' requires in_ch.")
            return gen_cls(in_ch=in_ch, num_frames=num_frames, **kwargs)

        if method == 'warp':
            if in_ch is None:
                raise ValueError(f"Method '{method}' requires in_ch.")
            return gen_cls(in_ch=in_ch, num_frames=num_frames, **kwargs)

        return gen_cls(num_frames=num_frames, **kwargs)


class GenericSequenceGenerator(nn.Module):
    """Interpolate two [B, C, H, W] inputs into a sequence of shape [B, C, T, H, W]."""
    def __init__(self, interp_method='linear', in_ch=None, num_frames=4, **kwargs):
        super().__init__()
        self.interp_method = interp_method
        self.num_frames = num_frames
        self.seq_generator = SequenceGeneratorFactory.build(
            interp_method,
            in_ch=in_ch,
            num_frames=num_frames,
            **kwargs
        )

    def forward(self, x1, x2):
        return self.seq_generator(x1, x2)


class BidirectionalResNet3D_v2_Demo(nn.Module):
    """Demonstrate sequence-generator integration; this is not a complete change model."""
    def __init__(self, in_ch=128, num_frames=4, interp_method='linear', **interp_kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.interp_method = interp_method
        self.seq_generator = SequenceGeneratorFactory.build(
            interp_method,
            in_ch=in_ch,
            num_frames=num_frames,
            **interp_kwargs
        )

    def generate_sequence(self, start, end, num_frames=None, return_alphas=False):
        if num_frames is None:
            num_frames = self.num_frames

        # Rebuild the generator when the requested sequence length changes.
        if num_frames != self.num_frames:
            temp_gen = SequenceGeneratorFactory.build(
                self.interp_method,
                in_ch=start.shape[1],
                num_frames=num_frames
            )
            seq = temp_gen(start, end)
        else:
            seq = self.seq_generator(start, end)

        if return_alphas:
            alphas = torch.linspace(0, 1, steps=num_frames, device=start.device, dtype=start.dtype)
            return seq, alphas
        return seq

    def forward(self, x1, x2):
        seq, alphas = self.generate_sequence(x1, x2, return_alphas=True)
        return seq, alphas
