
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib.colors import ListedColormap
from model.semseg.deeplabv3plus import DeepLabV3Plus  # Ensure this import works correctly

# Cityscapes color-to-index mapping
cityscapes_colors = {
    (128, 64, 128): 0,   # road
    (244, 35, 232): 1,   # sidewalk
    (70, 70, 70): 2,     # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170, 30): 6,   # traffic light
    (220, 220, 0): 7,    # traffic sign
    (107, 142, 35): 8,   # vegetation
    (152, 251, 152): 9,  # terrain
    (70, 130, 180): 10,  # sky
    (220, 20, 60): 11,   # person
    (255, 0, 0): 12,     # rider
    (0, 0, 142): 13,     # car
    (0, 0, 70): 14,      # truck
    (0, 60, 100): 15,    # bus
    (0, 80, 100): 16,    # train
    (0, 0, 230): 17,     # motorcycle
    (119, 11, 32): 18    # bicycle
}

# Set device to 'cuda' if GPU is available, otherwise fall back to 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def load_model(checkpoint_path, device):
    # Initialize the model and move it to the device
    model = DeepLabV3Plus(backbone='resnet101', nclass=19).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if 'model_state_dict' key exists, otherwise use the checkpoint directly
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # Assume the checkpoint is the state_dict directly

    # Load the state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    return model

def convert_color_to_index(label_path):
    label_image = Image.open(label_path).convert('RGB')
    label_array = np.array(label_image)
    label_indices = np.full(label_array.shape[:2], 255, dtype=np.uint8)  # Initialize with ignore_index (255)

    for color, class_index in cityscapes_colors.items():
        mask = np.all(label_array == color, axis=-1)
        label_indices[mask] = class_index

    return torch.from_numpy(label_indices).long()

def visualize_cross_entropy_map(model, image_path, label_path, device):
    # Define transformations for the image
    transform = transforms.Compose([
        transforms.Resize((321, 321)),
        transforms.ToTensor()
    ])

    # Load and preprocess the image
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # Convert label to class indices
    label = convert_color_to_index(label_path).to(device)  # Ensure the label is on the GPU

    # Ensure the label tensor is of type uint8 and convert to NumPy array
    label_numpy = label.cpu().numpy().astype(np.uint8)

    # Convert label tensor to PIL Image for resizing
    label_pil = Image.fromarray(label_numpy)

    # Resize the label using nearest neighbor interpolation
    label_resized = transforms.Resize((321, 321), interpolation=transforms.InterpolationMode.NEAREST)(label_pil)

    # Convert the resized label back to a tensor and move to the device
    label_tensor = torch.from_numpy(np.array(label_resized)).long().unsqueeze(0).to(device)  # Shape: (1, H, W)

    with torch.no_grad():
        # Get model prediction
        prediction = model(image)  # Shape: (N, C, H', W')

        # Upsample prediction to match label size
        prediction = F.interpolate(prediction, size=label_tensor.shape[-2:], mode='bilinear', align_corners=False)

        # Compute cross-entropy loss
        cross_entropy_loss = F.cross_entropy(prediction, label_tensor, ignore_index=255, reduction='none')

    return cross_entropy_loss.cpu().numpy().squeeze()


def overlay_entropy_map(image_path, entropy_map):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(entropy_map.shape[::-1], Image.LANCZOS)
    plt.imshow(image, alpha=0.6)
    plt.imshow(entropy_map, cmap='inferno', alpha=0.4)
    plt.colorbar(label="Entropy")
    plt.title("Input Image with Entropy Map")
    plt.axis('off')

def generate_pseudo_label(entropy_map, threshold=1.5):
    # Generate a binary pseudo-label based on the entropy threshold
    pseudo_label = np.zeros_like(entropy_map, dtype=np.uint8)
    pseudo_label[entropy_map < threshold] = 1
    return pseudo_label

def plot_pseudo_label(pseudo_label):
    # Plot the pseudo-label using a simple colormap
    cmap = ListedColormap(['white', 'teal'])
    plt.imshow(pseudo_label, cmap=cmap)
    plt.title("Pseudo-label after Filtering")
    plt.axis('off')

def plot_class_confidences(probabilities, class_names, location, title):
    # Extract and plot class confidences at the specified location
    h, w = probabilities.shape[1:]  # Get height and width
    x, y = location

    # Ensure the coordinates are within the valid range
    if x >= w or y >= h:
        print(f"Location {location} is out of bounds for the image of size ({w}, {h}).")
        return

    confidences = probabilities[:, y, x].cpu().numpy()
    plt.barh(class_names, confidences)
    plt.xlabel("Confidence")
    plt.title(title)

def create_figure(image_path, label_path, model, class_names, device='cuda'):
    # Generate the entropy map
    entropy_map = visualize_cross_entropy_map(model, image_path, label_path, device)
    pseudo_label = generate_pseudo_label(entropy_map)

    # Set up the plot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot (a): Input image with entropy map overlay
    plt.subplot(2, 2, 1)
    overlay_entropy_map(image_path, entropy_map)

    # Plot (b): Pseudo-label image
    plt.subplot(2, 2, 2)
    plot_pseudo_label(pseudo_label)

    # Get probabilities from model output for bar plots
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image)
        probabilities = F.softmax(prediction, dim=1).squeeze()

    # Plot (c): Reliable prediction
    plt.subplot(2, 2, 3)
    reliable_location = (100, 100)  # Replace with actual reliable point coordinates
    plot_class_confidences(probabilities, class_names, reliable_location, "Reliable Prediction")

    # Plot (d): Unreliable prediction
    plt.subplot(2, 2, 4)
    unreliable_location = (150, 150)  # Replace with actual unreliable point coordinates
    plot_class_confidences(probabilities, class_names, unreliable_location, "Unreliable Prediction")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cityscapes entropy-map visualization')
    parser.add_argument('--checkpoint', required=True, help='Path to a trained checkpoint (.pth)')
    parser.add_argument('--image', required=True, help='Path to an input image')
    parser.add_argument('--label', required=True, help='Path to the (color) ground-truth label')
    args = parser.parse_args()

    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        'motorcycle', 'bicycle'
    ]
    model = load_model(args.checkpoint, device)
    create_figure(args.image, args.label, model, class_names, device=device)


