import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

def symmetric_kl(p, q, eps=1e-8):
    # p and q are probability distributions with shape [B, C, H, W].
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=1)
    return kl_pq + kl_qp

class AdaptiveChangeCriterion(nn.Module):
    """Learn a per-image dissimilarity threshold from semantic logits [B, C, H, W].
    Use the resulting soft gate to supervise change logits [B, 1, H, W],
    enforce unchanged-region consistency, and separate changed regions."""
    def __init__(self,
                 temperature=0.15,      # Lower temperatures make the gate closer to a hard threshold.
                 margin=0.30,           # Minimum desired dissimilarity in changed regions.
                 w_bce=1.0,
                 w_same=0.5,
                 w_diff=0.5,
                 exclude_bg_idx=None    # Optionally exclude the background channel from similarity computation.
                 ):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(1.0))
        self.b_raw = nn.Parameter(torch.tensor(0.5))
        self.temperature_raw = nn.Parameter(torch.tensor(temperature))

        self.margin = margin
        self.w_bce = w_bce
        self.w_same = w_same
        self.w_diff = w_diff
        self.exclude_bg_idx = exclude_bg_idx

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits_a, logits_b, out_change):
        """Return the total loss and scalar component logs."""
        B, C, H, W = logits_a.shape

        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)

        # Exclude background to prevent it from dominating semantic similarity.
        if self.exclude_bg_idx is not None:
            keep = [i for i in range(C) if i != self.exclude_bg_idx]
            probs_a = probs_a[:, keep, :, :]
            probs_b = probs_b[:, keep, :, :]
            C_eff = len(keep)
        else:
            C_eff = C

        pa = probs_a.view(B, C_eff, -1)
        pb = probs_b.view(B, C_eff, -1)
        dot = torch.sum(probs_a * probs_b, dim=1)
        na = torch.sqrt(torch.sum(probs_a * probs_a, dim=1) + 1e-8)
        nb = torch.sqrt(torch.sum(probs_b * probs_b, dim=1) + 1e-8)
        cos_sim = dot / (na * nb + 1e-8)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        d = 1.0 - cos_sim

        # Compute one adaptive threshold per image, then broadcast over pixels.
        d_flat = d.view(B, -1)
        d_mean = d_flat.mean(dim=1, keepdim=True)
        d_std  = d_flat.std(dim=1,  keepdim=True)

        a = F.softplus(self.a_raw)
        b = F.softplus(self.b_raw)
        tau = a.view(1,1) * d_mean + b.view(1,1) * d_std
        tau = torch.clamp(tau, 0.0, 1.5)

        T = F.softplus(self.temperature_raw) + 1e-6
        gate = torch.sigmoid((d - tau.view(B,1,1)) / T)

        # Use the differentiable gate as a soft target for the change branch.
        pc = out_change
        bce_map = self.bce(pc, gate.unsqueeze(1))
        loss_bce = bce_map.mean()

        # Enforce semantic consistency in unchanged regions.
        skl = symmetric_kl(F.softmax(logits_a, dim=1), F.softmax(logits_b, dim=1))
        loss_same = ( (1.0 - gate) * skl ).mean()

        # Require changed regions to reach the dissimilarity margin.
        diff_gap = F.relu(self.margin - d)
        loss_diff = (gate * diff_gap).mean()

        loss = self.w_bce * loss_bce + self.w_same * loss_same + self.w_diff * loss_diff

        with torch.no_grad():
            tau_scalar = tau.mean().item()
            gate_mean = gate.mean().item()
            d_mean_scalar = d_mean.mean().item()

        logs = {
            'loss_bce': float(loss_bce.item()),
            'loss_same': float(loss_same.item()),
            'loss_diff': float(loss_diff.item()),
            'tau': tau_scalar,
            'gate_mean': gate_mean,
            'd_mean': d_mean_scalar,
            'T': float(T.item()),
            'a': float(a.item()),
            'b': float(b.item()),
        }
        return loss, logs


def change_loss(out_change, labels_bn, use_focal=True, alpha=0.5, gamma=2.0, dice_w=0.5):
    if use_focal:
        prob = torch.sigmoid(out_change)
        ce = F.binary_cross_entropy_with_logits(out_change, labels_bn, reduction='none')
        p_t = prob*labels_bn + (1-prob)*(1-labels_bn)
        focal = ce * (1-p_t).pow(gamma)
        focal = (alpha*labels_bn + (1-alpha)*(1-labels_bn)) * focal
        ce_term = focal.mean()
    else:
        pos = labels_bn.sum().clamp_min(1.0)
        neg = (1-labels_bn).sum().clamp_min(1.0)
        pos_weight = (neg/pos).detach()
        ce_term = F.binary_cross_entropy_with_logits(out_change, labels_bn, pos_weight=pos_weight)
    probs = torch.sigmoid(out_change)
    num = 2*(probs*labels_bn).sum(dim=(1,2,3))
    den = (probs+labels_bn).sum(dim=(1,2,3)).clamp_min(1e-6)
    dice = 1 - (num/den).mean()
    return ce_term + dice_w * dice


def dice_loss(pred, target, smooth=1e-6):

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    union = torch.clamp(union, min=smooth)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice
def lc_dice_loss(dice_A, dice_B):
    softmax = nn.Softmax(dim=1)

    dice_A = softmax(dice_A)
    dice_B = softmax(dice_B)

    dice_bg = dice_loss(dice_A[:, 0], dice_B[:, 0])
    # Swap appearance and disappearance channels between temporal directions.
    dice_fg1 = dice_loss(dice_A[:, 1], dice_B[:, 2])
    dice_fg2 = dice_loss(dice_A[:, 2], dice_B[:, 1])

    lc_dice = dice_bg + dice_fg1 + dice_fg2

    return lc_dice


class ChangeAlignmentLoss(nn.Module):
    def __init__(self):
        super(ChangeAlignmentLoss, self).__init__()

    def forward(self, outputs_A, outputs_B):
        outputs_A = F.softmax(outputs_A, dim=1)
        outputs_B = F.softmax(outputs_B, dim=1)
        pred_A = torch.argmax(outputs_A, dim=1)
        pred_B = torch.argmax(outputs_B, dim=1)
        mask_A = (pred_A == 0).float()
        mask_B = (pred_B == 0).float()

        loss = F.binary_cross_entropy(mask_A, mask_B, reduction='mean')
        return loss

def similarity_loss(output1, output2, labels_bn):

    similarity = F.cosine_similarity(output1, output2, dim=1)
    similarity_min = similarity.amin(dim=(1,2), keepdim=True)
    similarity_max = similarity.amax(dim=(1,2), keepdim=True)
    similarity = (similarity - similarity_min) / (similarity_max - similarity_min + 1e-8)
    binary_map = torch.sigmoid((0.5 - similarity) * 5)
    loss = F.binary_cross_entropy(binary_map, labels_bn.squeeze(1), reduction='mean')
    return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='elementwise_mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def CrossEntropy2d(input, target, weight=None, size_average=False):
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss

def weighted_BCE(output, target, weight_pos=None, weight_neg=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)

    if weight_pos is not None:
        loss = weight_pos * (target * torch.log(output)) + \
               weight_neg * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy_with_logits(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    # Normalize positive and negative contributions separately to balance classes.
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        loss = -((1-pt)**self.gamma) * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class ChangeSimilarity(nn.Module):
    """Apply pixelwise cosine embedding loss to semantic logits using a binary change mask."""
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        # CosineEmbeddingLoss uses +1 for unchanged pixels and -1 for changed pixels.
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])

        loss = self.loss_f(x1, x2, target)
        return loss

class ChangeSalience(nn.Module):
    """Match background probabilities with symmetric stop-gradient MSE."""
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.MSELoss(reduction=reduction)

    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)[:,0,:,:]
        x2 = F.softmax(x2, dim=1)[:,0,:,:]

        loss = self.loss_f(x1, x2.detach()) + self.loss_f(x2, x1.detach())
        return loss*0.5


def pix_loss(output, target, pix_weight, ignore_index=None):
    if ignore_index is not None:
        active_pos = 1-(target==ignore_index).unsqueeze(1).cuda().float()
        pix_weight *= active_pos

    batch_size, _, H, W = output.size()
    logp = F.log_softmax(output, dim=1)
    logp = logp.gather(1, target.view(batch_size, 1, H, W))
    # Normalize by the total pixel weight to keep the loss scale stable.
    weighted_logp = (logp * pix_weight).view(batch_size, -1)
    weighted_loss = weighted_logp.sum(1) / pix_weight.view(batch_size, -1).sum(1)
    weighted_loss = -1.0 * weighted_loss.mean()
    return weighted_loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        target = F.one_hot(target, num_classes=predict.shape[1]).permute(0, 3, 1, 2)

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
