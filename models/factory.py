"""Select the model and dataset-specific settings used for checkpoint validation."""

TINY_INTERNAL_SIZES = {'second': 512, 'landsat': 384}


def build_model(variant, dataset, dinov3_pretrained_path=None, resnet_pretrained=False):
    from dataset_process.datasets import DATASETS
    options = dict(num_classes=len(DATASETS[dataset]['classes']),
                   dinov3_pretrained_path=dinov3_pretrained_path,
                   resnet_pretrained=resnet_pretrained)
    if variant == 'tiny':
        from .FVT_Tiny import FVT_Tiny
        model_class = FVT_Tiny
        options['internal_size'] = TINY_INTERNAL_SIZES[dataset]
    elif variant == 'full':
        from .FVT import FVT
        model_class = FVT
    else:
        raise ValueError(f'Unknown model variant: {variant}')
    return model_class(**options)
