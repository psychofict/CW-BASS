#############################
#       Import Modules      #
#############################
# Local module imports
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

# Standard library imports
import argparse
from copy import deepcopy
from itertools import cycle
import os
import pickle
import logging
import yaml

# Third-party imports
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

#############################
#       Global Settings     #
#############################
MODE = None
torch.backends.cudnn.benchmark = True

#############################
#       Argument Parsing    #
#############################
def parse_args():
    parser = argparse.ArgumentParser(description='CW-BASS: Confidence-Weighted Boundary-Aware SSSS')

    # Basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'], default='deeplabv3plus')
    parser.add_argument("--resume-from", type=str, default=None, help="Path to resume model from")

    # Semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    # CW-BASS hyper-parameters
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Confidence-weighting exponent in the weighted CE loss')
    parser.add_argument('--base-threshold', type=float, default=0.6,
                        help='Base value for the dynamic confidence threshold')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Sensitivity of the dynamic threshold to mean confidence')
    parser.add_argument('--decay-factor', type=float, default=0.9,
                        help='Multiplicative confidence-decay factor')
    parser.add_argument('--use-confidence-decay', action='store_true',
                        help='Enable progressive confidence decay')
    parser.add_argument('--lambda-u', type=float, default=1.0,
                        help='Weight of the unsupervised (pseudo-label) loss term')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Epochs over which the unsupervised weight is linearly warmed up')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay for the mean-teacher model')

    # Retraining specific arguments
    parser.add_argument('--reliable-id-path', type=str)

    # Optional YAML config + checkpoint resume
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config (configs/*.yaml) to override defaults')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')

    args = parser.parse_args()

    # If a YAML config is supplied, use it to fill in values not set on the command line.
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader) or {}
        for key in ('epochs', 'crop_size', 'lr'):
            if getattr(args, key) is None and cfg.get(key) is not None:
                setattr(args, key, cfg[key])

    return args

#############################
#   Loss and Utility Func   #
#############################
def compute_pixel_confidence(prediction):
    """Pixel-wise confidence = max softmax probability. Returns [B, H, W]."""
    probs = F.softmax(prediction, dim=1)
    return torch.max(probs, dim=1).values

def weighted_cross_entropy_loss(prediction, pseudo_labels, confidence, gamma=1.0, ignore_index=255):
    """Confidence-weighted cross-entropy, averaged over non-ignored pixels only."""
    ce_loss = F.cross_entropy(prediction, pseudo_labels, reduction='none', ignore_index=ignore_index)
    valid = (pseudo_labels != ignore_index).float()
    weighted = (confidence ** gamma) * ce_loss * valid
    return weighted.sum() / valid.sum().clamp(min=1.0)

def decay_confidence(confidence, decay_factor=0.9, use_confidence_decay=False):
    """Optionally decays the confidence weights."""
    return confidence * decay_factor if use_confidence_decay else confidence

def dynamic_thresholding(confidence, base_threshold=0.6, beta=0.5, min_threshold=0.3, max_threshold=0.8):
    """Confidence threshold that adapts to the batch's mean confidence."""
    avg_conf = confidence.mean()
    threshold = base_threshold / (1 + torch.exp(-beta * (avg_conf - 0.5)))
    return torch.clamp(threshold, min=min_threshold, max=max_threshold)

def detect_boundaries(labels, num_classes):
    """Sobel boundary mask from a label map. Returns [B, H, W] in {0, 1}.

    `num_classes` is passed explicitly (rather than inferred from labels.max())
    so an ignore label (255) can never blow up the one-hot dimension.
    """
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=labels.device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=labels.device, dtype=torch.float32)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)

    edges_x = F.conv2d(one_hot, sobel_x, padding=1, groups=num_classes)
    edges_y = F.conv2d(one_hot, sobel_y, padding=1, groups=num_classes)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return (edges.sum(dim=1) > 0).float()

def boundary_loss(pred, pseudo_labels, confidence, boundary_mask, gamma=1.0, ignore_index=255):
    """Confidence-weighted CE plus an extra CE term on boundary pixels.

    Both terms ignore pixels labelled `ignore_index` (below-threshold pseudo-labels).
    """
    base = weighted_cross_entropy_loss(pred, pseudo_labels, confidence, gamma, ignore_index)

    ce = F.cross_entropy(pred, pseudo_labels, reduction='none', ignore_index=ignore_index)
    valid_boundary = boundary_mask * (pseudo_labels != ignore_index).float()
    boundary_term = (ce * valid_boundary).sum() / valid_boundary.sum().clamp(min=1.0)
    return base + 0.5 * boundary_term

#############################
#  Training and Validation  #
#############################
@torch.no_grad()
def update_ema(teacher, student, decay):
    """In-place EMA update of the teacher's parameters and buffers from the student."""
    s = student.module if isinstance(student, DataParallel) else student
    t = teacher.module if isinstance(teacher, DataParallel) else teacher
    for t_param, s_param in zip(t.parameters(), s.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
    for t_buf, s_buf in zip(t.buffers(), s.buffers()):
        t_buf.data.copy_(s_buf.data)


def train_semi_supervised(student, teacher, labeled_loader, unlabeled_loader, optimizer,
                          criterion, nclass, epoch, args, device, base_lrs, total_iters):
    """One epoch of confidence-weighted, boundary-aware semi-supervised training.

    Supervised CE on the labeled batch (using its ground-truth mask) is combined
    with an unsupervised confidence-weighted boundary loss on the unlabeled batch,
    whose targets are online pseudo-labels from the EMA teacher, filtered by a
    dynamic confidence threshold. The unsupervised weight is linearly warmed up
    over the first ``args.warmup_epochs`` epochs. Learning rate follows a poly
    schedule applied per parameter group.
    """
    student.train()
    teacher.eval()  # stable pseudo-labels from EMA'd weights/buffers; updated only via update_ema

    n_iters = max(len(labeled_loader), len(unlabeled_loader))
    labeled_iter, unlabeled_iter = iter(cycle(labeled_loader)), iter(cycle(unlabeled_loader))
    warmup_iters = args.warmup_epochs * n_iters
    epoch_loss = 0.0

    for it in range(n_iters):
        global_iter = epoch * n_iters + it
        lr_scale = (1 - global_iter / total_iters) ** 0.9
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * lr_scale

        l_img, l_mask = next(labeled_iter)
        u_img, _ = next(unlabeled_iter)
        l_img, l_mask, u_img = l_img.to(device), l_mask.to(device), u_img.to(device)

        # ----- supervised term (ground-truth labels) -----
        sup_loss = criterion(student(l_img), l_mask)

        # ----- teacher pseudo-labels on unlabeled images (online) -----
        with torch.no_grad():
            t_logits = teacher(u_img)
            confidence = compute_pixel_confidence(t_logits)
            pseudo = torch.argmax(t_logits, dim=1)
            boundary_mask = detect_boundaries(pseudo, nclass)
            threshold = dynamic_thresholding(confidence, args.base_threshold, args.beta)
            # pixels below the dynamic threshold are ignored, not forced to background
            pseudo = torch.where(confidence > threshold, pseudo, torch.full_like(pseudo, 255))
            confidence = decay_confidence(confidence, args.decay_factor, args.use_confidence_decay)

        # ----- unsupervised confidence-weighted boundary term -----
        unsup_loss = boundary_loss(student(u_img), pseudo, confidence, boundary_mask, args.gamma)

        ramp = min(1.0, global_iter / warmup_iters) if warmup_iters > 0 else 1.0
        loss = sup_loss + args.lambda_u * ramp * unsup_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(teacher, student, args.ema_decay)

        epoch_loss += loss.item()
        if it % 50 == 0:
            print(f"  iter {it}/{n_iters}  loss {loss.item():.4f} "
                  f"(sup {sup_loss.item():.4f}, unsup {unsup_loss.item():.4f}, "
                  f"lr {optimizer.param_groups[0]['lr']:.2e})")

    return epoch_loss / max(n_iters, 1)

def train(model, trainloader, valloader, criterion, optimizer, args, device, accumulation_steps=4):
    """
    General training loop with gradient accumulation and learning rate scheduling.
    Validates and saves the best model based on mIoU.
    """
    iters, best_miou = 0, 0.0
    total_iters = len(trainloader) * args.epochs
    best_model = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        print(f"\n==> Epoch {epoch + 1}/{args.epochs}")

        for i, (img, mask) in enumerate(tqdm(trainloader)):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trainloader):.4f}")

        current_miou = validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device)
        if current_miou > best_miou:
            best_miou = current_miou
            best_model = deepcopy(model)

    return best_model if best_model else model

def validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device):
    """
    Validates the model on the validation set, computes the mean IoU,
    and saves a checkpoint if a new best is achieved.
    """
    model.eval()
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(valloader, desc="Validating"):
            if len(batch) == 2:
                img, mask = batch
            elif len(batch) >= 3:
                img, mask, *_ = batch

            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = criterion(logits, mask)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            metric.add_batch(preds.cpu().numpy(), mask.cpu().numpy())

    mIOU = metric.evaluate()[-1] * 100
    print(f"Validation mIoU: {mIOU:.2f}%")

    if mIOU > best_miou:
        save_checkpoint(epoch, model, optimizer, mIOU, os.path.join(args.save_path, "best_checkpoint.pth"))
        print(f"New best mIoU: {mIOU:.2f}%")
        return mIOU

    return best_miou

#############################
#   Checkpoint Functions    #
#############################
def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a checkpoint and updates the model and optimizer.
    Adjusts keys if DataParallel is used.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    print(f"Loading checkpoint from '{checkpoint_path}'")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("Checkpoint loaded with weights_only=True")
    except (TypeError, pickle.UnpicklingError):
        print("weights_only=True failed. Re-loading with weights_only=False.")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state_dict = model.state_dict()
    updated_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not next(iter(model_state_dict)).startswith('module.'):
            updated_state_dict[k[len('module.'):]] = v
        elif not k.startswith('module.') and next(iter(model_state_dict)).startswith('module.'):
            updated_state_dict['module.' + k] = v
        else:
            updated_state_dict[k] = v

    model.load_state_dict(updated_state_dict, strict=False)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_miou = checkpoint.get('best_miou', 0.0)
    print(f"Resumed training from epoch {epoch}, best mIoU: {best_miou:.2f}")
    return epoch + 1, best_miou

def save_checkpoint(epoch, model, optimizer, mIOU, path):
    """
    Saves a checkpoint containing the model state, optimizer state, and best mIoU.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mIOU': mIOU
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch}, mIOU: {mIOU:.2f}")

#############################
#   Retraining Pipeline     #
#############################
def select_reliable(models, dataloader, args):
    """
    Selects reliable image IDs based on pairwise IoU between model predictions.
    Splits the IDs into reliable and unreliable files.
    """
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for model in models:
        model.eval()
    id_to_reliability = []

    with torch.no_grad():
        for img, mask, img_id in tqdm(dataloader, desc="Selecting reliable IDs"):
            img = img.cuda()
            preds = [torch.argmax(model(img), dim=1).cpu().numpy() for model in models]
            mious = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
                metric.add_batch(preds[i], preds[-1])
                mious.append(metric.evaluate()[-1])
            reliability = sum(mious) / len(mious) if mious else 0.0
            id_to_reliability.append((img_id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    half = len(id_to_reliability) // 2
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:half]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[half:]:
            f.write(elem[0] + '\n')

def label(model, dataloader, args):
    """
    Generates pseudo-labels for images using the trained model and saves them.
    """
    model.eval()
    tbar = tqdm(dataloader, desc="Generating Pseudo-Labels")
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)

    with torch.no_grad():
        for img, mask, img_id in tbar:
            img = img.cuda()
            pred = model(img)
            pred_labels = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred_labels.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1] * 100

            pred_image = Image.fromarray(pred_labels.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_image.putpalette(cmap)
            save_path = os.path.join(args.pseudo_mask_path, os.path.basename(img_id[0].split(' ')[1]))
            pred_image.save(save_path)

            tbar.set_description(f'Generating pseudo-labels - mIoU: {mIOU:.2f}%')

def retrain_on_pseudo_labeled_data(args, unlabeled_path, best_model, device, valloader, criterion):
    """
    Retrains the model on a mix of labeled and pseudo-labeled data.
    """
    MODE = 'semi_train'
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, unlabeled_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args, device)
    best_model = train(model, trainloader, valloader, CrossEntropyLoss(ignore_index=255), optimizer, args, device)
    return best_model

def pseudo_label_and_retrain(args, best_model, device, valloader, criterion):
   
    # Stage A: Select Reliable IDs
    print('\n================> A: Select Reliable IDs')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    select_reliable([best_model], dataloader, args)

    # Stage B: Pseudo Label Reliable Images
    print('\n================> B: Pseudo Label Reliable Images')
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

    # Stage C: Pseudo Label Reliable Images
    print('\n================> C: Pseudo Label Reliable Images')
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

#############################
#  Model Initialization     #
#############################
def init_basic_elems(args, device):
    """
    Initializes the segmentation model and optimizer.
    Loads a pretrained backbone if applicable and converts BatchNorm layers for multi-GPU training.
    """
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)

    # NOTE: DataParallel (used below) is incompatible with SyncBatchNorm, which
    # requires a torch.distributed process group. Plain BatchNorm is used.

    # Load pretrained weights for the backbone if available
    if args.model == 'deeplabv3plus' and args.backbone in ['resnet50', 'resnet101']:
        weight_path = f'pretrained/{args.backbone}.pth'
        if os.path.isfile(weight_path):
            backbone_state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
            model.backbone.load_state_dict(backbone_state_dict, strict=False)
        else:
            print(f"[warn] pretrained backbone not found at '{weight_path}'; training from scratch.")

    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr * 10.0}
    ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    return model, optimizer

#############################
#        Main Routine       #
#############################
if __name__ == '__main__':
    args = parse_args()

    # Set default parameters based on the dataset
    dataset_defaults = {
        'pascal': {'epochs': 80, 'lr': 0.001, 'crop_size': 321},
        'cityscapes': {'epochs': 240, 'lr': 0.004, 'crop_size': 721}
    }
    defaults = dataset_defaults.get(args.dataset, {})
    args.epochs = args.epochs or defaults.get('epochs')
    args.lr = args.lr or (defaults.get('lr') / 16 * args.batch_size)
    args.crop_size = args.crop_size or defaults.get('crop_size')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Arguments: {args}")

    # Create the save directory if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255).to(device)

    # Initialize datasets and dataloaders
    nclass = 21 if args.dataset == 'pascal' else 19
    num_workers = min(8, os.cpu_count() or 1)

    labeled_set = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path)
    unlabeled_set = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.unlabeled_id_path)
    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1, shuffle=False,
                           num_workers=4, pin_memory=True, drop_last=False)

    # Student model + optimizer, and an EMA (mean-teacher) copy with no gradients
    model, optimizer = init_basic_elems(args, device)
    teacher = deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"\nModel parameters: {count_params(model):.1f}M")

    base_lrs = [group['lr'] for group in optimizer.param_groups]
    iters_per_epoch = max(len(labeled_loader), len(unlabeled_loader))
    total_iters = iters_per_epoch * args.epochs

    start_epoch, best_miou = 0, 0.0
    if args.resume:
        start_epoch, best_miou = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from epoch {start_epoch}, best mIoU: {best_miou:.2f}")

    # Semi-supervised training loop (confidence-weighted, boundary-aware, mean-teacher)
    for epoch in range(start_epoch, args.epochs):
        print(f"\n==> Epoch {epoch + 1}/{args.epochs}")
        avg_loss = train_semi_supervised(model, teacher, labeled_loader, unlabeled_loader, optimizer,
                                         criterion, nclass, epoch, args, device, base_lrs, total_iters)
        print(f"Epoch {epoch + 1} mean loss: {avg_loss:.4f}")
        best_miou = validate_and_checkpoint(model, valloader, criterion, optimizer, epoch, best_miou, args, device)

    print("\nTraining completed. Best validation mIoU: %.2f%%" % best_miou)

