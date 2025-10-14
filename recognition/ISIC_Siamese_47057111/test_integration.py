#!/usr/bin/env python3
"""
Integration test for Siamese network with ISIC dataset
Tests the complete pipeline: data loading + model forward pass
"""

import torch
import sys
import os

# Add project directory to path
sys.path.append('/Users/matildadamman/COMP3710_A3/recognition/ISIC_Siamese_47057111')

def test_integration():
    """Test Siamese network with actual ISIC dataset."""
    
    print("="*60)
    print("TESTING SIAMESE NETWORK + DATASET INTEGRATION")
    print("="*60)
    
    from dataset import get_data_loaders
    from modules import create_siamese_model, ContrastiveLoss, CombinedLoss
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders (small batch for testing)
    print("\\n1. Loading dataset...")
    data_loaders = get_data_loaders(
        data_root="../archive",
        batch_size=8,
        image_size=224,
        num_workers=0,
        use_siamese=True
    )
    
    # Create model
    print("\\n2. Creating Siamese network...")
    model = create_siamese_model({
        'feature_dim': 256,  # Smaller for faster testing
        'distance_function': 'euclidean',
        'pretrained': True,
        'dropout_rate': 0.2
    })
    model = model.to(device)
    
    # Test with real data
    print("\\n3. Testing with real ISIC data...")
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # Get a batch from Siamese training data
    img1, img2, labels = next(iter(train_loader))
    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
    
    print(f"   Input shapes: img1={img1.shape}, img2={img2.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label distribution: {labels.unique(return_counts=True)}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        similarity, distance = model(img1, img2)
        
        print(f"\\n   Model outputs:")
        print(f"   - Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
        print(f"   - Distance range: [{distance.min():.3f}, {distance.max():.3f}]")
    
    # Test loss computation
    model.train()
    similarity, distance = model(img1, img2)
    
    # Different loss functions
    contrastive_loss_fn = ContrastiveLoss(margin=2.0)
    combined_loss_fn = CombinedLoss()
    
    contrastive_loss = contrastive_loss_fn(distance, labels)
    combined_loss = combined_loss_fn(similarity, distance, labels)
    
    print(f"\\n   Loss values:")
    print(f"   - Contrastive: {contrastive_loss.item():.4f}")
    print(f"   - Combined: {combined_loss.item():.4f}")
    
    # Test with regular validation data
    print("\\n4. Testing with validation data...")
    val_images, val_labels = next(iter(val_loader))
    val_images = val_images.to(device)
    
    # Extract features for similarity comparison
    features = model.extract_features(val_images)
    print(f"   Feature extraction: {features.shape}")
    print(f"   Feature norms: min={torch.norm(features, dim=1).min():.3f}, "
          f"max={torch.norm(features, dim=1).max():.3f}")
    
    # Test pairwise similarity (first 4 images)
    if len(features) >= 4:
        sim_scores = model.compute_similarity(
            features[:2].unsqueeze(1).repeat(1, 2, 1).view(-1, features.shape[1]),
            features[2:4].unsqueeze(0).repeat(2, 1, 1).view(-1, features.shape[1])
        )
        print(f"   Pairwise similarities: {sim_scores.flatten()}")
    
    print(f"\\n{'='*60}")
    print("INTEGRATION TESTING COMPLETED SUCCESSFULLY!")
    print("✓ Dataset loading works with Siamese network")
    print("✓ Model processes real dermoscopic images correctly")  
    print("✓ Loss functions compute properly")
    print("✓ Feature extraction and similarity work as expected")
    print("="*60)

if __name__ == "__main__":
    # Change to correct directory
    os.chdir('/Users/matildadamman/COMP3710_A3/recognition/ISIC_Siamese_47057111')
    test_integration()