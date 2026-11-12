"""Train/evaluate the active FVT model on SECOND or Landsat-SCD."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from dataset_process.datasets import ChangeDataset, DATASETS
from utils.loss import CrossEntropyLoss2d, ChangeSimilarity, weighted_BCE_logits
from utils.utils import SCDD_eval_all


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='second')
    parser.add_argument('--model', choices=['full', 'tiny'], default='full')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dinov3-weights', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/second'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=3, help='Per process/GPU')
    parser.add_argument('--val-batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-power', type=float, default=1.0)
    parser.add_argument('--weight-decay', type=float, default=1e-7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--checkpoint', type=Path, help='Model state_dict for evaluation')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--split', default='val', help='Evaluation split')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.val_batch_size, args.print_freq) < 1 or args.workers < 0:
        parser.error('Epochs, batch sizes and print frequency must be positive; workers must be nonnegative')
    if args.eval_only and args.checkpoint is None:
        parser.error('--eval-only requires --checkpoint')
    if not args.eval_only and args.dinov3_weights is None:
        parser.error('Training requires --dinov3-weights (full: ViT-L/16; tiny: ViT-S/16)')
    return args


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(loader, model, criterion, device, num_classes):
    model.eval()
    predictions, labels = [], []
    loss_sum, count, accuracy_sum = 0.0, 0, 0.0
    for a, b, la, lb in loader:
        a, b, la, lb = [x.to(device) for x in (a, b, la, lb)]
        change, sa, sb, _ = model(a, b, labels_bn=(la > 0).unsqueeze(1).float())
        # An unchanged-only target has no supervised semantic pixels (class 0
        # is ignored); mean CE would otherwise return NaN for this batch.
        loss_a = criterion(sa, la) if (la != 0).any() else sa.new_zeros(())
        loss_b = criterion(sb, lb) if (lb != 0).any() else sb.new_zeros(())
        loss = (loss_a + loss_b) * 0.5
        mask = (change.sigmoid() > 0.5).squeeze(1)
        pa, pb = sa.argmax(1) * mask, sb.argmax(1) * mask
        for pred, target in [(pa, la), (pb, lb)]:
            predictions.extend(pred.cpu().numpy())
            labels.extend(target.cpu().numpy())
            accuracy_sum += (pred == target).float().flatten(1).mean(1).sum().item()
        loss_sum += loss.item() * a.size(0)
        count += a.size(0)
    fscd, miou, sek = SCDD_eval_all(predictions, labels, num_classes)
    return {'loss': loss_sum / count, 'Fscd': float(fscd), 'mIoU': float(miou),
            'Sek': float(sek), 'OA': accuracy_sum / (2 * count)}


def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; CPU evaluation can use --device cpu')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if distributed:
        dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
    seed_all(args.seed)
    from models.factory import build_model
    spec = DATASETS[args.dataset]
    model = build_model(args.model, args.dataset,
                    dinov3_pretrained_path=str(args.dinov3_weights) if args.dinov3_weights else None,
                    resnet_pretrained=not args.eval_only).to(device)
    # Original model constructor fixes its initialization seed to 42.
    seed_all(args.seed)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    criterion = CrossEntropyLoss2d(ignore_index=0).to(device)
    val_set = ChangeDataset(args.data_root, args.split, args.dataset)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=args.workers)
    if args.eval_only:
        if rank == 0:
            print(json.dumps(evaluate(val_loader, model, criterion, device, len(spec['classes'])), indent=2))
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    train_set = ChangeDataset(args.data_root, 'train', args.dataset, augment=True)
    sampler = DistributedSampler(train_set, seed=args.seed, drop_last=True) if distributed else None
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, num_workers=args.workers, drop_last=True,
                              worker_init_fn=seed_worker, generator=generator)
    if not len(train_loader):
        raise ValueError('Training set is smaller than the total batch size')
    if distributed:
        model = DDP(model, device_ids=[local_rank] if args.device == 'cuda' else None,
                    find_unused_parameters=True)
    raw_model = model.module if distributed else model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(0.9, 0.99), eps=1e-8,
                                  weight_decay=args.weight_decay)
    similarity = ChangeSimilarity().to(device)
    writer = None
    if rank == 0:
        from tensorboardX import SummaryWriter
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
        writer = SummaryWriter(str(args.output_dir / 'logs'))
    total_steps = len(train_loader) * args.epochs
    best_miou = -float('inf')
    history = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for index, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + index + 1
            lr = args.lr * (1.0 - step / total_steps) ** args.lr_power
            for group in optimizer.param_groups:
                group['lr'] = lr
            a, b, la, lb = [x.to(device) for x in batch]
            change_target = (la > 0).unsqueeze(1).float()
            optimizer.zero_grad()
            change, sa, sb, _ = model(a, b, change_target)
            seg = (criterion(sa, la) + criterion(sb, lb)) * 0.5
            binary = weighted_BCE_logits(change, change_target)
            sc = similarity(sa[:, 1:], sb[:, 1:], change_target)
            loss = seg + binary + sc
            loss.backward()
            optimizer.step()
            if rank == 0 and step % args.print_freq == 0:
                print(f'Epoch {epoch + 1}/{args.epochs} step {index + 1}/{len(train_loader)} loss={loss.item():.4f} lr={lr:.6g}', flush=True)
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/lr', lr, step)
        if distributed:
            dist.barrier()
        # Full validation on rank 0 avoids padded validation samples and concurrent output writes.
        if rank == 0:
            metrics = evaluate(val_loader, raw_model, criterion, device, len(spec['classes']))
            print(f'Epoch {epoch + 1}: {metrics}', flush=True)
            history.append({'epoch': epoch + 1, **metrics})
            (args.output_dir / 'metrics.json').write_text(json.dumps(history, indent=2))
            for name, value in metrics.items():
                writer.add_scalar('val/' + name, value, epoch + 1)
            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                torch.save(raw_model.state_dict(), args.output_dir / 'best.pth')
        if distributed:
            dist.barrier()
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
