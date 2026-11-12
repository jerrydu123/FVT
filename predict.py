"""Predict semantic change maps for matching PNG pairs without ground truth labels."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dataset_process.datasets import DATASETS, image_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--im1-dir', type=Path, required=True)
    parser.add_argument('--im2-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS, default='second')
    parser.add_argument('--model', choices=['full', 'tiny'], default='full')
    parser.add_argument('--output-dir', type=Path, default=Path('predictions'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    from models.factory import build_model
    spec = DATASETS[args.dataset]
    model = build_model(args.model, args.dataset, resnet_pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    model.to(args.device).eval()
    files = sorted(args.im1_dir.glob('*.png'))
    if not files:
        raise ValueError(f'No PNG images in {args.im1_dir}')
    for folder in ['change', 'semantic1', 'semantic2', 'semantic1_rgb', 'semantic2_rgb']:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    palette = np.asarray(spec['palette'], dtype=np.uint8)
    with torch.no_grad():
        for path in files:
            pair = args.im2_dir / path.name
            with Image.open(path) as image:
                a = np.array(image.convert('RGB'))
            with Image.open(pair) as image:
                b = np.array(image.convert('RGB'))
            if a.shape != b.shape:
                raise ValueError(f'Pair sizes do not match: {path.name}')
            a, b = [image_tensor(image, spec, time).unsqueeze(0).to(args.device)
                    for time, image in enumerate([a, b])]
            change, sa, sb, _ = model(a, b)
            mask = (change.sigmoid() > 0.5).squeeze(1)
            binary = mask[0].cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(binary).save(args.output_dir / 'change' / path.name)
            for index, semantic in enumerate([sa, sb], 1):
                pred = (semantic.argmax(1) * mask)[0].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred).save(args.output_dir / f'semantic{index}' / path.name)
                Image.fromarray(palette[pred]).save(args.output_dir / f'semantic{index}_rgb' / path.name)
            print(path.name, flush=True)


if __name__ == '__main__':
    main()
