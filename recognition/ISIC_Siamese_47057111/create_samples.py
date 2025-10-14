"""
Generate sample visualizations for README documentation
Creates sample images showing dataset structure and Siamese pairs
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_data_loaders, ISICDataset, get_transforms
import os
import sys

def create_sample_visualization():
    """Create sample data visualization for README"""
    
    print("Creating sample visualization...")
    
    # Set up paths
    data_root = "../archive"
    
    # Create data loaders (non-Siamese for individual samples)
    data_loaders = get_data_loaders(
        data_root=data_root,
        batch_size=8,
        image_size=224,
        num_workers=0,
        use_siamese=False
    )
    
    # Get a batch from validation set
    val_loader = data_loaders['val']
    images, labels = next(iter(val_loader))
    
    # Create figure for individual samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('ISIC 2020 Dataset Samples', fontsize=16, fontweight='bold')
    
    # Denormalization parameters (reverse ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Denormalize image
        img = images[i].numpy().transpose(1, 2, 0)
        img = (img * std) + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{"Melanoma" if labels[i].item() == 1 else "Normal"}', 
                                fontweight='bold',
                                color='red' if labels[i].item() == 1 else 'green')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create Siamese pairs visualization
    siamese_loaders = get_data_loaders(
        data_root=data_root,
        batch_size=4,
        image_size=224,
        num_workers=0,
        use_siamese=True
    )
    
    train_loader = siamese_loaders['train']
    img1_batch, img2_batch, pair_labels = next(iter(train_loader))
    
    # Create figure for Siamese pairs
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Siamese Network Training Pairs', fontsize=16, fontweight='bold')
    
    for i in range(4):
        # Denormalize images
        img1 = img1_batch[i].numpy().transpose(1, 2, 0)
        img1 = (img1 * std) + mean
        img1 = np.clip(img1, 0, 1)
        
        img2 = img2_batch[i].numpy().transpose(1, 2, 0)
        img2 = (img2 * std) + mean
        img2 = np.clip(img2, 0, 1)
        
        # Display pair
        axes[0, i].imshow(img1)
        axes[0, i].set_title(f'Image 1 (Pair {i+1})', fontweight='bold')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img2)
        axes[1, i].set_title(f'Image 2 (Pair {i+1})', fontweight='bold')
        axes[1, i].axis('off')
        
        # Add pair label information
        pair_type = "Same Class" if pair_labels[i].item() == 1 else "Different Class"
        color = 'green' if pair_labels[i].item() == 1 else 'red'
        
        # Add text annotation between the images
        fig.text(0.125 + i * 0.2, 0.02, pair_type, 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('siamese_pairs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated dataset_samples.png")
    print("✓ Generated siamese_pairs.png")

if __name__ == "__main__":
    # Change to correct directory
    os.chdir('/Users/matildadamman/COMP3710_A3/recognition/ISIC_Siamese_47057111')
    create_sample_visualization()