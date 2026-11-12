"""Paired RGB images and RGB/index semantic change labels."""
from pathlib import Path
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from utils.transform import rand_rot90_flip_MCD

DATASETS = {
    'second': {
        'classes': ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field'],
        'palette': [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]],
        'mean': [[113.40,114.08,116.45], [111.07,114.04,118.18]],
        'std': [[48.30,46.27,48.14], [49.41,47.01,47.94]],
    },
    'landsat': {
        'classes': ['unchanged', 'farmland', 'desert', 'building', 'water'],
        'palette': [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]],
        'mean': [[141.53,139.20,137.73], [137.36,136.50,135.144]],
        'std': [[81.99,83.31,83.89], [85.97,86.01,86.81]],
    },
}


def image_tensor(image, spec, time):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an RGB image with three channels')
    image = (image - np.asarray(spec['mean'][time])) / np.asarray(spec['std'][time])
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def label_indices(label, spec):
    if label.ndim == 3 and label.shape[2] == 3:
        result = np.full(label.shape[:2], -1, dtype=np.int64)
        for index, color in enumerate(spec['palette']):
            result[np.all(label == color, axis=-1)] = index
        label = result
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError('Labels must be RGB palette images or integer class-index images')
    if label.min() < 0 or label.max() >= len(spec['classes']):
        raise ValueError('Unknown label colors or out-of-range class indices')
    return label.astype(np.int64)


class ChangeDataset(Dataset):
    def __init__(self, root, split, dataset='second', augment=False):
        self.root = Path(root) / split
        self.spec = DATASETS[dataset]
        self.augment = augment
        self.names = sorted(p.name for p in (self.root / 'im1').glob('*.png'))
        if not self.names:
            raise ValueError(f'No PNG images found in {self.root / "im1"}')
        for folder in ['im2', 'label1', 'label2']:
            missing = [name for name in self.names if not (self.root / folder / name).is_file()]
            if missing:
                raise FileNotFoundError(f'Missing paired file: {self.root / folder / missing[0]}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        a, b = [io.imread(self.root / folder / name) for folder in ['im1', 'im2']]
        la, lb = [label_indices(io.imread(self.root / folder / name), self.spec)
                  for folder in ['label1', 'label2']]
        if not (a.shape[:2] == b.shape[:2] == la.shape == lb.shape):
            raise ValueError(f'Image/label sizes do not match for {name}')
        if self.augment:
            a, b, la, lb = rand_rot90_flip_MCD(a, b, la, lb)
        return (image_tensor(a, self.spec, 0), image_tensor(b, self.spec, 1),
                torch.from_numpy(np.ascontiguousarray(la)),
                torch.from_numpy(np.ascontiguousarray(lb)))
