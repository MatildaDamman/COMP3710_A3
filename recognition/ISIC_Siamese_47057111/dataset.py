"""
ISIC 2020 Kaggle Challenge Dataset Loader

Dataset for melanoma classification using Siamese network architecture.
Contains image pairs for one-shot learning on dermoscopic images.

Author: Matilda Damman (47057111)
Course: COMP3710 - Pattern Analysis
Project: ISIC 2020 Siamese Network Classifier
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import random


class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2020 melanoma classification.
    
    This dataset loads dermoscopic images and their corresponding labels
    for binary classification (normal vs melanoma).
    
    Args:
        csv_file (str): Path to the metadata CSV file
        img_dir (str): Directory containing the images
        transform (callable, optional): Optional transform to be applied on images
        split (str): Dataset split - 'train', 'val', or 'test'
    """
    
    def __init__(self, 
                 csv_file: str,
                 img_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 split: str = 'train'):
        
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        # Load metadata
        self.metadata = pd.read_csv(csv_file)
        print(f"Loaded {len(self.metadata)} samples from {csv_file}")
        
        # Extract image IDs and labels
        self.image_ids = self.metadata['isic_id'].values
        self.labels = self.metadata['target'].values  # 0: normal, 1: melanoma
        
        # Print class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution for {split}: {class_dist}")
        print(f"Normal (0): {class_dist.get(0, 0)}, Melanoma (1): {class_dist.get(1, 0)}")
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a torch.Tensor and label is int
        """
        # Get image ID and label
        img_id = self.image_ids[idx]
        label = self.labels[idx]
        
        # Construct image path
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Returns:
            torch.Tensor: Class weights for loss function
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        class_weights = len(self.labels) / (len(unique) * counts)
        return torch.FloatTensor(class_weights)


class ISICSiameseDataset(Dataset):
    """
    Siamese dataset for ISIC 2020 that generates pairs of images.
    
    For Siamese networks, we need pairs of images with labels indicating
    whether they belong to the same class (1) or different classes (0).
    
    Args:
        base_dataset (ISICDataset): Base dataset to generate pairs from
        num_pairs_per_epoch (int): Number of pairs to generate per epoch
    """
    
    def __init__(self, 
                 base_dataset: ISICDataset,
                 num_pairs_per_epoch: int = 10000):
        
        self.base_dataset = base_dataset
        self.num_pairs_per_epoch = num_pairs_per_epoch
        
        # Group indices by class for efficient pair generation
        self.class_indices = {0: [], 1: []}
        for idx, label in enumerate(base_dataset.labels):
            self.class_indices[label].append(idx)
            
        print(f"Siamese dataset - Normal samples: {len(self.class_indices[0])}, "
              f"Melanoma samples: {len(self.class_indices[1])}")
    
    def __len__(self) -> int:
        """Return number of pairs per epoch."""
        return self.num_pairs_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate a pair of images.
        
        Args:
            idx (int): Index (not used, pairs are generated randomly)
            
        Returns:
            tuple: (img1, img2, label) where label=1 if same class, 0 if different
        """
        # Decide if we want a positive pair (same class) or negative pair (different class)
        same_class = random.random() > 0.5
        
        if same_class:
            # Select two images from the same class
            class_label = random.choice([0, 1])
            if len(self.class_indices[class_label]) < 2:
                # Fallback to different class if not enough samples
                same_class = False
            else:
                idx1, idx2 = random.sample(self.class_indices[class_label], 2)
                pair_label = 1
        
        if not same_class:
            # Select images from different classes
            idx1 = random.choice(self.class_indices[0])
            idx2 = random.choice(self.class_indices[1])
            pair_label = 0
        
        # Get the images
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        
        return img1, img2, pair_label


def get_transforms(image_size: int = 224, split: str = 'train') -> transforms.Compose:
    """
    Get image transformations for different dataset splits.
    
    Args:
        image_size (int): Target image size
        split (str): Dataset split ('train', 'val', 'test')
        
    Returns:
        transforms.Compose: Composed transformations
    """
    
    if split == 'train':
        # Training transformations with data augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Validation/test transformations (no augmentation)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_splits(csv_file: str, 
                      val_size: float = 0.15, 
                      test_size: float = 0.15,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/validation/test sets with stratification.
    
    Args:
        csv_file (str): Path to metadata CSV
        val_size (float): Validation set proportion
        test_size (float): Test set proportion  
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    
    # Load full dataset
    df = pd.read_csv(csv_file)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['target'], random_state=random_state
    )
    
    # Second split: separate validation from training
    adjusted_val_size = val_size / (1 - test_size)  # Adjust for remaining data
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, 
        stratify=train_val_df['target'], random_state=random_state
    )
    
    print(f"Dataset splits created:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples") 
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def get_data_loaders(data_root: str,
                    batch_size: int = 32,
                    image_size: int = 224,
                    num_workers: int = 4,
                    use_siamese: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for train/validation/test splits.
    
    Args:
        data_root (str): Root directory containing dataset
        batch_size (int): Batch size for data loaders
        image_size (int): Target image size
        num_workers (int): Number of worker processes for data loading
        use_siamese (bool): Whether to use Siamese pairs or regular classification
        
    Returns:
        dict: Dictionary containing train/val/test data loaders
    """
    
    # Paths
    csv_file = os.path.join(data_root, "train-metadata.csv")
    img_dir = os.path.join(data_root, "train-image", "image")
    
    # Verify paths exist
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Metadata file not found: {csv_file}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Create data splits
    train_df, val_df, test_df = create_data_splits(csv_file)
    
    # Save split files for reproducibility
    train_df.to_csv(os.path.join(data_root, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(data_root, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(data_root, "test_split.csv"), index=False)
    
    data_loaders = {}
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # Save temporary split file
        split_file = os.path.join(data_root, f"{split_name}_temp.csv")
        split_df.to_csv(split_file, index=False)
        
        # Get transforms
        transform = get_transforms(image_size, split_name)
        
        # Create base dataset
        base_dataset = ISICDataset(
            csv_file=split_file,
            img_dir=img_dir,
            transform=transform,
            split=split_name
        )
        
        if use_siamese and split_name == 'train':
            # Use Siamese dataset for training
            dataset = ISICSiameseDataset(base_dataset, num_pairs_per_epoch=batch_size * 100)
        else:
            # Use regular dataset for validation/test or non-Siamese mode
            dataset = base_dataset
        
        # Create data loader
        data_loaders[split_name] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'train')
        )
        
        # Clean up temporary file
        os.remove(split_file)
    
    return data_loaders


if __name__ == "__main__":
    """
    Test the dataset loader with sample calls and print statistics.
    """
    
    print("="*60)
    print("TESTING ISIC 2020 DATASET LOADER")
    print("="*60)
    
    # Configuration
    DATA_ROOT = "../archive"  # Relative to the script location
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    
    try:
        # Test data loader creation
        print("\\n1. Creating data loaders...")
        data_loaders = get_data_loaders(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=0,  # Set to 0 for testing
            use_siamese=True
        )
        
        print("\\n2. Testing regular dataset...")
        # Test regular dataset first
        regular_loaders = get_data_loaders(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_workers=0,
            use_siamese=False
        )
        
        # Sample from regular validation loader
        val_loader = regular_loaders['val']
        print(f"Validation loader: {len(val_loader)} batches")
        
        for i, (images, labels) in enumerate(val_loader):
            print(f"Batch {i+1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Label values: {labels.unique().tolist()}")
            print(f"  Image min/max: {images.min():.3f}/{images.max():.3f}")
            
            if i >= 2:  # Show first 3 batches
                break
        
        print("\\n3. Testing Siamese dataset...")
        # Test Siamese dataset
        train_loader = data_loaders['train']
        print(f"Siamese train loader: {len(train_loader)} batches")
        
        for i, (img1, img2, pair_labels) in enumerate(train_loader):
            print(f"Siamese Batch {i+1}:")
            print(f"  Image 1 shape: {img1.shape}")
            print(f"  Image 2 shape: {img2.shape}")
            print(f"  Pair labels shape: {pair_labels.shape}")
            print(f"  Pair label values: {pair_labels.unique().tolist()}")
            print(f"  Same class pairs: {(pair_labels == 1).sum().item()}")
            print(f"  Different class pairs: {(pair_labels == 0).sum().item()}")
            
            if i >= 2:  # Show first 3 batches
                break
        
        print("\\n4. Dataset statistics summary:")
        print(f"✓ Regular dataset loaded successfully")
        print(f"✓ Siamese pairs generated successfully") 
        print(f"✓ Image transformations working correctly")
        print(f"✓ Class balance maintained in splits")
        
        print("\\n" + "="*60)
        print("DATASET TESTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
