# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib.colors import ListedColormap
from model.semseg.deeplabv3plus import DeepLabV3Plus


def load_model_from_checkpoint(checkpoint_path, model_class, device='cuda'):
    # Initialize model
    model = model_class(backbone='resnet101', nclass=21).to(device)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)  # Use 'model_state_dict' if available

    # Match state_dict keys
    new_state_dict = {}
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.startswith('module.') and not next(iter(model_state_dict)).startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        elif not k.startswith('module.') and next(iter(model_state_dict)).startswith('module.'):
            new_state_dict['module.' + k] = v  # Add 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load state_dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    return model


def overlay_predictions(image_path, predicted_mask, class_names):
    # Load the input image
    image = Image.open(image_path).convert('RGB')

    # Create a color map for visualization
    num_classes = len(class_names)
    colors = plt.cm.get_cmap('tab20', num_classes)

    # Overlay the segmentation mask on the image
    plt.figure(figsize=(20, 10))
    plt.imshow(image, alpha=0.6)  # Input image with some transparency
    plt.imshow(predicted_mask, cmap=colors, alpha=0.4)  # Predicted segmentation mask
    plt.colorbar(ticks=range(num_classes), label="Classes")
    plt.title("Input Image with Predicted Mask")
    plt.axis('off')
    plt.show()


def generate_predictions(model, image_path, device='cuda'):
    # Preprocess the input image
    transform = transforms.Compose([transforms.ToTensor()])  # No resizing
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # Generate model predictions
    with torch.no_grad():
        prediction = model(image)
        predicted_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    return predicted_mask


def plot_comparison(image_path, predicted_mask, ground_truth_path, class_names):
    # Load the input image
    image = Image.open(image_path).convert('RGB')

    # Load the ground truth mask
    ground_truth = np.array(Image.open(ground_truth_path))

    # Create a color map
    num_classes = len(class_names)
    colors = plt.cm.get_cmap('tab20', num_classes)

    # Plot the input image, ground truth, and prediction
    plt.figure(figsize=(20, 10))

    # Subplot 1: Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # Subplot 2: Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap=colors)
    plt.title("Ground Truth")
    plt.axis('off')

    # Subplot 3: Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)  # Semi-transparent overlay
    plt.imshow(predicted_mask, cmap=colors, alpha=0.4)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pascal VOC ground-truth vs prediction visualization')
    parser.add_argument('--checkpoint', required=True, help='Path to a trained checkpoint (.pth)')
    parser.add_argument('--image', required=True, help='Path to an input image')
    parser.add_argument('--ground-truth', required=True, help='Path to the ground-truth mask')
    args = parser.parse_args()

    # Pascal VOC class names
    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_checkpoint(args.checkpoint, DeepLabV3Plus, device)
    predicted_mask = generate_predictions(model, args.image, device)
    overlay_predictions(args.image, predicted_mask, class_names)
    plot_comparison(args.image, predicted_mask, args.ground_truth, class_names)
