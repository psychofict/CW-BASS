
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from matplotlib.colors import ListedColormap
from model.semseg.deeplabv3plus import DeepLabV3Plus  # Import the model class

def load_model_from_checkpoint(checkpoint_path, model_class, device='cpu'):
    # Initialize the model
    model = model_class(backbone='resnet101', nclass=21).to(device)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)  # Use 'model_state_dict' if available

    # Adjust the state_dict keys to match the model structure
    new_state_dict = {}
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.startswith('module.') and not next(iter(model_state_dict)).startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        elif not k.startswith('module.') and next(iter(model_state_dict)).startswith('module.'):
            new_state_dict['module.' + k] = v  # Add 'module.' prefix
        else:
            new_state_dict[k] = v  # Keep the key as is

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    return model

def visualize_cross_entropy_map(model, image_path, label_path, device='cpu'):
    # Image transform (scaled to [0,1] tensor).
    img_transform = transforms.Compose([
        transforms.Resize((321, 321)),  # Adjust size based on model crop
        transforms.ToTensor()
    ])
    image = img_transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # The label holds integer class indices (0..20, 255=ignore). It must NOT be
    # run through ToTensor (which would rescale by 1/255); resize with NEAREST
    # and keep the raw indices as a LongTensor.
    label_pil = Image.open(label_path).resize((321, 321), Image.NEAREST)
    label = torch.from_numpy(np.array(label_pil)).long().unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image)
        prediction = F.interpolate(prediction, size=label.shape[-2:], mode='bilinear', align_corners=False)
        cross_entropy_loss = F.cross_entropy(prediction, label, ignore_index=255, reduction='none')

    # Move loss to CPU for visualization and return it
    return cross_entropy_loss.cpu().numpy().squeeze()

def overlay_entropy_map(image_path, entropy_map):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(entropy_map.shape[::-1], Image.LANCZOS)
    plt.imshow(image, alpha=0.6)
    plt.imshow(entropy_map, cmap='inferno', alpha=0.4)
    plt.colorbar(label="Entropy")
    plt.scatter([100], [100], color='yellow', marker='+', s=100)  # Replace with actual coordinates
    plt.scatter([150], [150], color='white', marker='+', s=100)   # Replace with actual coordinates
    plt.title("Input image with entropy map.")

def generate_pseudo_label(entropy_map, threshold=1.5):
    pseudo_label = np.zeros_like(entropy_map)
    pseudo_label[entropy_map < threshold] = 1  # Assuming binary, can adjust for multiple classes
    return pseudo_label

def plot_pseudo_label(pseudo_label):
    cmap = ListedColormap(['white', 'teal'])  # Adjust colors for each class
    plt.imshow(pseudo_label, cmap=cmap)
    plt.title("Pseudo-label after filtering.")

def generate_pseudo_mask(model, image_path, threshold=0.6, device='cpu'):
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((321, 321)),
        transforms.ToTensor()
    ])

    # Load and preprocess the image
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    # Get model predictions
    with torch.no_grad():
        prediction = model(image)
        probabilities = F.softmax(prediction, dim=1).squeeze().cpu().numpy()

    # Generate the pseudo mask by selecting the class with the highest probability if above the threshold
    pseudo_mask = np.argmax(probabilities, axis=0)
    confidence_mask = np.max(probabilities, axis=0)
    pseudo_mask[confidence_mask < threshold] = 0  # Set to background if confidence is below threshold

    return pseudo_mask

def plot_pseudo_mask(pseudo_mask, class_names):
    # Define a colormap with specific colors for each class
    cmap_colors = [
        'black',       # 0: background
        'red',         # 1: aeroplane
        'blue',        # 2: bicycle
        'green',       # 3: bird
        'cyan',        # 4: boat
        'yellow',      # 5: bottle
        'magenta',     # 6: bus
        'orange',      # 7: car
        'purple',      # 8: cat
        'brown',       # 9: chair
        'lime',        # 10: cow
        'pink',        # 11: diningtable
        'gold',        # 12: dog
        'navy',        # 13: horse
        'violet',      # 14: motorbike
        'salmon',      # 15: person
        'olive',       # 16: potted plant
        'turquoise',   # 17: sheep
        'maroon',      # 18: sofa
        'chocolate',   # 19: train
        'plum'         # 20: tv/monitor
    ]

    # Ensure the colormap matches the number of classes
    cmap = ListedColormap(cmap_colors[:len(class_names)])

    # Plot the pseudo mask
    plt.imshow(pseudo_mask, cmap=cmap)
    plt.colorbar(ticks=range(len(class_names)), label="Classes")
    plt.title("Pseudo Mask Based on Model Predictions")



def plot_class_confidences(probabilities, class_names, location, title):
    confidences = probabilities[:, location[1], location[0]].cpu().numpy()  # Extract confidence at location
    plt.barh(class_names, confidences)
    plt.xlabel("Confidence")
    plt.title(title)

def create_figure(image_path, label_path, model, class_names, device='cpu'):
    # Generate entropy map
    entropy_map = visualize_cross_entropy_map(model, image_path, label_path, device)
    pseudo_label = generate_pseudo_label(entropy_map)
    pseudo_mask = generate_pseudo_mask(model, image_path, threshold=0.6)

    # Set up the plot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
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

    # Plot (c): Pseudo mask based on model predictions
    plt.subplot(2, 2, 3)
    plot_pseudo_mask(pseudo_mask, class_names)

    # Get probabilities from model output for bar plots
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image)
        probabilities = F.softmax(prediction, dim=1).squeeze()

    # Plot (d): Reliable prediction
    plt.subplot(2, 2, 4)
    reliable_location = (100, 100)  # Replace with actual reliable point coordinates
    plot_class_confidences(probabilities, class_names, reliable_location, "Reliable Prediction")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pascal VOC entropy-map visualization')
    parser.add_argument('--checkpoint', required=True, help='Path to a trained checkpoint (.pth)')
    parser.add_argument('--image', required=True, help='Path to an input image')
    parser.add_argument('--label', required=True, help='Path to the ground-truth mask')
    args = parser.parse_args()

    class_names = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_checkpoint(args.checkpoint, DeepLabV3Plus, device)
    create_figure(args.image, args.label, model, class_names, device)



