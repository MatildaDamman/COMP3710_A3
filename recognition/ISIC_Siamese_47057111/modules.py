"""
Siamese Network Architecture for ISIC 2020 Melanoma Classification

This module implements a Siamese neural network for binary skin lesion classification.
The network consists of twin CNN branches that learn to distinguish between
melanoma and normal skin lesions through contrastive learning.

Author: Matilda Damman (47057111)
Course: COMP3710 - Pattern Analysis
Project: ISIC 2020 Siamese Network Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import math


class FeatureExtractor(nn.Module):
    """
    Feature extraction backbone for Siamese network.
    
    Uses a pre-trained ResNet18 as the backbone CNN for extracting meaningful
    features from dermoscopic images. The final classification layer is removed
    to obtain feature vectors instead of class predictions.
    
    Args:
        pretrained (bool): Whether to use ImageNet pre-trained weights
        feature_dim (int): Dimension of output feature vectors
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self, 
                 pretrained: bool = True, 
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3):
        super(FeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Load pre-trained ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        # ResNet18 has 512 features before the final FC layer
        backbone_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom feature projection head
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Initialize weights for new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers using Xavier initialization."""
        for module in self.feature_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.
        
        Args:
            x (torch.Tensor): Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Feature vector [batch_size, feature_dim]
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Project to desired feature dimension
        features = self.feature_head(features)
        
        # L2 normalize features for stable distance computation
        features = F.normalize(features, p=2, dim=1)
        
        return features


class DistanceLayer(nn.Module):
    """
    Distance computation layer for Siamese networks.
    
    Computes various distance metrics between feature vectors from twin networks.
    Supports multiple distance functions for flexibility in training and evaluation.
    
    Args:
        distance_function (str): Type of distance function ('euclidean', 'cosine', 'manhattan')
    """
    
    def __init__(self, distance_function: str = 'euclidean'):
        super(DistanceLayer, self).__init__()
        
        self.distance_function = distance_function.lower()
        
        # Validate distance function
        valid_functions = ['euclidean', 'cosine', 'manhattan', 'learned']
        if self.distance_function not in valid_functions:
            raise ValueError(f"Distance function must be one of {valid_functions}")
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two feature vectors.
        
        Args:
            features1 (torch.Tensor): Features from first image [batch_size, feature_dim]
            features2 (torch.Tensor): Features from second image [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: Distance values [batch_size, 1]
        """
        if self.distance_function == 'euclidean':
            # Euclidean distance: ||f1 - f2||_2
            distance = torch.sqrt(torch.sum((features1 - features2) ** 2, dim=1, keepdim=True) + 1e-8)
            
        elif self.distance_function == 'cosine':
            # Cosine distance: 1 - cos_similarity(f1, f2)
            cos_sim = F.cosine_similarity(features1, features2, dim=1)
            distance = (1 - cos_sim).unsqueeze(1)
            
        elif self.distance_function == 'manhattan':
            # Manhattan distance: ||f1 - f2||_1
            distance = torch.sum(torch.abs(features1 - features2), dim=1, keepdim=True)
            
        else:  # learned distance
            # Element-wise absolute difference for learned distance
            distance = torch.abs(features1 - features2)
        
        return distance


class SimilarityHead(nn.Module):
    """
    Similarity prediction head for Siamese network.
    
    Takes distance/difference features and predicts whether two images
    belong to the same class (similarity score).
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout_rate: float = 0.3):
        super(SimilarityHead, self).__init__()
        
        self.similarity_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output similarity probability [0, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.similarity_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, distance_features: torch.Tensor) -> torch.Tensor:
        """
        Predict similarity from distance features.
        
        Args:
            distance_features (torch.Tensor): Distance/difference features
            
        Returns:
            torch.Tensor: Similarity probability [batch_size, 1]
        """
        return self.similarity_net(distance_features)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    Implements the contrastive loss function that pulls similar pairs closer
    and pushes dissimilar pairs apart by at least a margin.
    
    Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    Where:
    - Y: 1 if same class, 0 if different class
    - D: Distance between feature vectors
    - margin: Minimum distance for dissimilar pairs
    
    Args:
        margin (float): Margin for dissimilar pairs
        reduction (str): Loss reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, margin: float = 2.0, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, 
                distance: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            distance (torch.Tensor): Distance between feature pairs [batch_size, 1]
            label (torch.Tensor): Labels (1 for same class, 0 for different class) [batch_size]
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        # Ensure label is float and has correct shape
        label = label.float().view(-1, 1)
        
        # Contrastive loss computation
        similar_loss = (1 - label) * torch.pow(distance, 2)
        dissimilar_loss = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = 0.5 * (similar_loss + dissimilar_loss)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class SiameseNetwork(nn.Module):
    """
    Complete Siamese Network for ISIC 2020 melanoma classification.
    
    The network consists of:
    1. Twin feature extractors (shared weights)
    2. Distance computation layer
    3. Similarity prediction head
    
    Args:
        feature_dim (int): Dimension of feature vectors
        distance_function (str): Distance computation method
        pretrained (bool): Use pre-trained backbone
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self,
                 feature_dim: int = 512,
                 distance_function: str = 'euclidean',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super(SiameseNetwork, self).__init__()
        
        # Shared feature extractor for both input images
        self.feature_extractor = FeatureExtractor(
            pretrained=pretrained,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # Distance computation layer
        self.distance_layer = DistanceLayer(distance_function=distance_function)
        
        # Similarity prediction head
        if distance_function == 'learned':
            # For learned distance, input dimension is the feature difference
            similarity_input_dim = feature_dim
        else:
            # For computed distances, input is scalar distance
            similarity_input_dim = 1
        
        self.similarity_head = SimilarityHead(
            input_dim=similarity_input_dim,
            dropout_rate=dropout_rate
        )
        
        self.distance_function = distance_function
    
    def forward(self, 
                img1: torch.Tensor, 
                img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Siamese network.
        
        Args:
            img1 (torch.Tensor): First image [batch_size, 3, 224, 224]
            img2 (torch.Tensor): Second image [batch_size, 3, 224, 224]
            
        Returns:
            tuple: (similarity_score, distance)
            - similarity_score: Probability that images are from same class [batch_size, 1]
            - distance: Distance between feature vectors [batch_size, 1]
        """
        # Extract features from both images using shared network
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        
        # Compute distance between features
        distance = self.distance_layer(features1, features2)
        
        # Predict similarity
        similarity = self.similarity_head(distance)
        
        return similarity, distance
    
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single image (useful for inference).
        
        Args:
            img (torch.Tensor): Input image [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Feature vector [batch_size, feature_dim]
        """
        return self.feature_extractor(img)
    
    def compute_similarity(self, 
                          features1: torch.Tensor, 
                          features2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between pre-extracted features.
        
        Args:
            features1 (torch.Tensor): Features from first image
            features2 (torch.Tensor): Features from second image
            
        Returns:
            torch.Tensor: Similarity score
        """
        distance = self.distance_layer(features1, features2)
        similarity = self.similarity_head(distance)
        return similarity


class CombinedLoss(nn.Module):
    """
    Combined loss function for Siamese networks.
    
    Combines contrastive loss with binary cross-entropy loss for more stable training.
    
    Args:
        contrastive_weight (float): Weight for contrastive loss
        bce_weight (float): Weight for binary cross-entropy loss
        margin (float): Margin for contrastive loss
    """
    
    def __init__(self, 
                 contrastive_weight: float = 0.7,
                 bce_weight: float = 0.3,
                 margin: float = 2.0):
        super(CombinedLoss, self).__init__()
        
        self.contrastive_weight = contrastive_weight
        self.bce_weight = bce_weight
        
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, 
                similarity: torch.Tensor,
                distance: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            similarity (torch.Tensor): Predicted similarity scores
            distance (torch.Tensor): Computed distances
            labels (torch.Tensor): True labels (1 for same, 0 for different)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Contrastive loss on distances
        contrastive = self.contrastive_loss(distance, labels)
        
        # Binary cross-entropy loss on similarities
        bce = self.bce_loss(similarity.squeeze(), labels.float())
        
        # Weighted combination
        total_loss = (self.contrastive_weight * contrastive + 
                     self.bce_weight * bce)
        
        return total_loss


def create_siamese_model(config: Optional[dict] = None) -> SiameseNetwork:
    """
    Factory function to create Siamese network with default or custom configuration.
    
    Args:
        config (dict, optional): Model configuration parameters
        
    Returns:
        SiameseNetwork: Configured Siamese network model
    """
    
    # Default configuration
    default_config = {
        'feature_dim': 512,
        'distance_function': 'euclidean',
        'pretrained': True,
        'dropout_rate': 0.3
    }
    
    # Update with custom config if provided
    if config:
        default_config.update(config)
    
    # Create model
    model = SiameseNetwork(**default_config)
    
    return model


if __name__ == "__main__":
    """
    Test the Siamese network architecture with sample inputs.
    """
    
    print("="*60)
    print("TESTING SIAMESE NETWORK ARCHITECTURE")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_siamese_model()
    model = model.to(device)
    
    # Print model architecture
    print(f"\\nModel Architecture:")
    print(f"- Feature extractor parameters: {sum(p.numel() for p in model.feature_extractor.parameters()):,}")
    print(f"- Distance layer: {model.distance_layer.distance_function}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample inputs
    batch_size = 4
    img1 = torch.randn(batch_size, 3, 224, 224).to(device)
    img2 = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    print(f"\\nTesting with batch size: {batch_size}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        similarity, distance = model(img1, img2)
        
        print(f"\\nOutput shapes:")
        print(f"- Similarity: {similarity.shape}")
        print(f"- Distance: {distance.shape}")
        print(f"\\nOutput ranges:")
        print(f"- Similarity: [{similarity.min():.3f}, {similarity.max():.3f}]")
        print(f"- Distance: [{distance.min():.3f}, {distance.max():.3f}]")
    
    # Test loss computation
    model.train()
    similarity, distance = model(img1, img2)
    
    # Test different loss functions
    contrastive_loss = ContrastiveLoss()
    combined_loss = CombinedLoss()
    
    loss1 = contrastive_loss(distance, labels)
    loss2 = combined_loss(similarity, distance, labels)
    
    print(f"\\nLoss values:")
    print(f"- Contrastive loss: {loss1.item():.4f}")
    print(f"- Combined loss: {loss2.item():.4f}")
    
    # Test feature extraction
    features1 = model.extract_features(img1)
    features2 = model.extract_features(img2)
    
    print(f"\\nFeature extraction:")
    print(f"- Feature shape: {features1.shape}")
    print(f"- Feature norm (should be ~1.0): {torch.norm(features1, dim=1).mean():.3f}")
    
    # Test similarity computation
    sim_score = model.compute_similarity(features1, features2)
    print(f"- Similarity from features: {sim_score.shape}")
    
    print(f"\\n{'='*60}")
    print("SIAMESE NETWORK TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)
