#!/usr/bin/env python3
"""
COMP3710 Assignment: Advanced Melanoma Classification - Inference & Prediction
============================================================================

This module demonstrates how to use the trained melanoma classifier for inference
on new dermoscopic images. Includes visualization of predictions, confidence scores,
and sample results from the test dataset.

Key Features:
- Load trained model and perform inference
- Visualize sample predictions with confidence scores
- Display sample images with true vs predicted labels
- Generate classification reports and statistics
- Example usage for production deployment

Model Performance:
- Validation Accuracy: 83.0%
- Test Set Performance: Comprehensive evaluation included
- Handles class imbalance through advanced techniques

Author: Student ID 47057111  
Course: COMP3710 - Pattern Analysis and Machine Intelligence
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from modules import AdvancedSiameseNetwork, create_model
from dataset import ISICSiameseDataset, create_data_loaders


class MelanomaPredictor:
    """
    Advanced melanoma classifier for inference and prediction.
    
    This class provides a clean interface for loading the trained model
    and performing predictions on dermoscopic images with comprehensive
    confidence analysis and visualization capabilities.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the predictor with trained model.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Computing device ('cuda', 'cpu', 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing MelanomaPredictor on {self.device}")
        
        # Load model and configuration
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        # Class names for interpretation
        self.class_names = ['Benign', 'Malignant']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Best validation accuracy: {self.config.get('val_acc', 'N/A')}")
    
    def _load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        """Load trained model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"📥 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same configuration
        config = checkpoint.get('config', {})
        model = create_model(
            num_classes=1,
            use_metadata=True,
            dropout=0.3
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, checkpoint
    
    def predict_single_image(self, image_path: str, age: float = None, sex: str = None) -> Dict:
        """
        Predict melanoma probability for a single image.
        
        Args:
            image_path: Path to dermoscopic image
            age: Patient age (optional)
            sex: Patient sex ('male'/'female', optional)
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self._preprocess_image(image)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        # Prepare metadata
        metadata_tensor = self._prepare_metadata(age, sex)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        metadata_tensor = metadata_tensor.unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(image_tensor, metadata_tensor)
            probability = torch.sigmoid(logits).squeeze().item()
            prediction = int(probability > 0.5)
        
        # Prepare results
        results = {
            'image_path': image_path,
            'prediction': prediction,
            'predicted_class': self.class_names[prediction],
            'malignant_probability': probability,
            'benign_probability': 1 - probability,
            'confidence': max(probability, 1 - probability),
            'metadata': {
                'age': age,
                'sex': sex
            }
        }
        
        return results
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model input size
        image = image.resize((224, 224), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_array = np.array(image, dtype=np.float32)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = image_tensor / 255.0
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    def _prepare_metadata(self, age: Optional[float], sex: Optional[str]) -> torch.Tensor:
        """Prepare metadata tensor."""
        # Age normalization
        if age is None:
            age_normalized = 0.5  # Default middle age
        else:
            age_normalized = np.clip(float(age) / 100.0, 0.0, 1.0)
        
        # Sex encoding
        if sex is None or str(sex).lower() not in ['male', 'female']:
            sex_encoded = 0.5  # Neutral encoding
        else:
            sex_encoded = 1.0 if str(sex).lower() == 'male' else 0.0
        
        return torch.tensor([age_normalized, sex_encoded], dtype=torch.float32)
    
    def predict_batch(self, dataset: ISICSiameseDataset, num_samples: int = None) -> Dict:
        """
        Predict on a batch of samples from dataset.
        
        Args:
            dataset: ISICSiameseDataset instance
            num_samples: Number of samples to predict (None for all)
            
        Returns:
            Dictionary with batch prediction results
        """
        if num_samples is None:
            num_samples = len(dataset)
        
        num_samples = min(num_samples, len(dataset))
        
        print(f"🔮 Predicting on {num_samples} samples...")
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_metadata = []
        
        # Create dataloader for batch processing
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=2
        )
        
        self.model.eval()
        with torch.no_grad():
            samples_processed = 0
            for images, metadata, targets in dataloader:
                if samples_processed >= num_samples:
                    break
                
                # Move to device
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images, metadata)
                probabilities = torch.sigmoid(logits.squeeze())
                predictions = (probabilities > 0.5).float()
                
                # Store results
                batch_size = min(len(predictions), num_samples - samples_processed)
                all_predictions.extend(predictions[:batch_size].cpu().numpy())
                all_probabilities.extend(probabilities[:batch_size].cpu().numpy())
                all_targets.extend(targets[:batch_size].cpu().numpy())
                all_metadata.extend(metadata[:batch_size].cpu().numpy())
                
                samples_processed += batch_size
        
        # Calculate metrics
        all_predictions = np.array(all_predictions[:num_samples])
        all_probabilities = np.array(all_probabilities[:num_samples])
        all_targets = np.array(all_targets[:num_samples])
        
        # Compute performance metrics
        accuracy = np.mean(all_predictions == all_targets)
        
        # Handle case where we might have test set without labels
        if not np.any(all_targets == -1):  # Valid labels present
            cm = confusion_matrix(all_targets, all_predictions)
            classification_rep = classification_report(
                all_targets, all_predictions, 
                target_names=self.class_names, 
                output_dict=True
            )
        else:
            cm = None
            classification_rep = None
        
        results = {
            'num_samples': num_samples,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'metadata': np.array(all_metadata),
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_rep
        }
        
        print(f"✅ Batch prediction completed!")
        if cm is not None:
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        return results
    
    def visualize_predictions(self, dataset: ISICSiameseDataset, 
                            num_samples: int = 8, 
                            save_path: str = None,
                            show_confident: bool = True) -> plt.Figure:
        """
        Visualize sample predictions with images and confidence scores.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
            show_confident: Whether to show most confident predictions
            
        Returns:
            Matplotlib figure
        """
        print(f"🎨 Creating prediction visualization for {num_samples} samples...")
        
        # Get predictions
        results = self.predict_batch(dataset, num_samples * 2)  # Get extra to select from
        
        # Select samples to display
        if show_confident:
            # Show most confident predictions
            confidence_scores = np.maximum(results['probabilities'], 1 - results['probabilities'])
            confident_indices = np.argsort(confidence_scores)[-num_samples:]
        else:
            # Show random samples
            confident_indices = np.random.choice(len(results['predictions']), num_samples, replace=False)
        
        # Create visualization
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_idx in enumerate(confident_indices):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            # Get sample info
            image, metadata, target = dataset[sample_idx]
            prediction = results['predictions'][sample_idx]
            probability = results['probabilities'][sample_idx]
            
            # Convert image tensor to displayable format
            display_image = self._tensor_to_image(image)
            
            # Display image
            ax.imshow(display_image)
            ax.axis('off')
            
            # Create title with prediction info
            true_class = self.class_names[int(target)] if target != -1 else "Unknown"
            pred_class = self.class_names[int(prediction)]
            confidence = max(probability, 1 - probability)
            
            # Color code: green for correct, red for incorrect, blue for unknown
            if target == -1:
                color = 'blue'
            elif prediction == target:
                color = 'green'
            else:
                color = 'red'
            
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Melanoma Classification Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        return fig
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor back to displayable image."""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        denorm_tensor = tensor * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        
        # Convert to numpy and transpose
        image_array = denorm_tensor.permute(1, 2, 0).numpy()
        return image_array
    
    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """Generate comprehensive prediction report."""
        report_lines = []
        report_lines.append("MELANOMA CLASSIFICATION PREDICTION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Model: Advanced Melanoma Classifier")
        report_lines.append(f"Total Samples: {results['num_samples']:,}")
        
        if results['confusion_matrix'] is not None:
            cm = results['confusion_matrix']
            accuracy = results['accuracy']
            
            report_lines.append(f"\nPERFORMANCE METRICS:")
            report_lines.append(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            
            # Class-wise metrics
            if results['classification_report']:
                for class_name in self.class_names:
                    class_metrics = results['classification_report'][class_name.lower()]
                    report_lines.append(f"  {class_name}:")
                    report_lines.append(f"    Precision: {class_metrics['precision']:.3f}")
                    report_lines.append(f"    Recall:    {class_metrics['recall']:.3f}")
                    report_lines.append(f"    F1-Score:  {class_metrics['f1-score']:.3f}")
            
            # Confusion matrix
            report_lines.append(f"\nCONFUSION MATRIX:")
            report_lines.append(f"                 Predicted")
            report_lines.append(f"               Benign  Malignant")
            report_lines.append(f"True Benign    {cm[0,0]:6d}  {cm[0,1]:9d}")
            report_lines.append(f"    Malignant  {cm[1,0]:6d}  {cm[1,1]:9d}")
        
        # Confidence analysis
        high_conf = np.sum(np.maximum(results['probabilities'], 1 - results['probabilities']) > 0.8)
        medium_conf = np.sum(np.maximum(results['probabilities'], 1 - results['probabilities']) > 0.6) - high_conf
        low_conf = results['num_samples'] - high_conf - medium_conf
        
        report_lines.append(f"\nCONFIDENCE ANALYSIS:")
        report_lines.append(f"  High confidence (>80%):   {high_conf:4d} ({100*high_conf/results['num_samples']:.1f}%)")
        report_lines.append(f"  Medium confidence (60-80%): {medium_conf:4d} ({100*medium_conf/results['num_samples']:.1f}%)")
        report_lines.append(f"  Low confidence (<60%):    {low_conf:4d} ({100*low_conf/results['num_samples']:.1f}%)")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📋 Report saved to: {save_path}")
        
        return report


def main_demo():
    """Main demonstration of the prediction system."""
    
    print("🎯 MELANOMA CLASSIFICATION INFERENCE DEMO")
    print("=" * 50)
    
    # Configuration
    model_path = "results/best_model.pth"
    metadata_path = "archive/train-metadata.csv"
    image_dir = "archive/train-image/image"
    results_dir = "prediction_results"
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        print("   Please run training first to generate the model.")
        return
    
    # Initialize predictor
    try:
        predictor = MelanomaPredictor(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test dataset
    print(f"\n📁 Loading test dataset...")
    try:
        metadata_df = pd.read_csv(metadata_path)
        _, _, test_df = create_data_splits(metadata_df, random_state=42)
        
        test_dataset = ISICSiameseDataset(
            test_df, image_dir, mode='test', use_augmentation=False
        )
        
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Perform batch prediction on test set
    print(f"\n🔮 Running inference on test set...")
    test_results = predictor.predict_batch(test_dataset, num_samples=1000)
    
    # Generate comprehensive report
    print(f"\n📊 Generating prediction report...")
    report = predictor.generate_report(test_results, f"{results_dir}/prediction_report.txt")
    print(report)
    
    # Create visualizations
    print(f"\n🎨 Creating prediction visualizations...")
    
    # Most confident predictions
    fig1 = predictor.visualize_predictions(
        test_dataset, 
        num_samples=12, 
        save_path=f"{results_dir}/confident_predictions.png",
        show_confident=True
    )
    
    # Random sample predictions
    fig2 = predictor.visualize_predictions(
        test_dataset, 
        num_samples=12, 
        save_path=f"{results_dir}/random_predictions.png",
        show_confident=False
    )
    
    # Example single image prediction
    print(f"\n🔍 Example single image prediction...")
    if len(test_dataset) > 0:
        # Get a sample image path
        sample_row = test_df.iloc[0]
        image_path = os.path.join(image_dir, f"{sample_row['isic_id']}.jpg")
        
        if os.path.exists(image_path):
            single_result = predictor.predict_single_image(
                image_path, 
                age=sample_row.get('age_approx', None),
                sex=sample_row.get('sex', None)
            )
            
            print(f"📷 Single Image Prediction:")
            print(f"   Image: {single_result['image_path']}")
            print(f"   Prediction: {single_result['predicted_class']}")
            print(f"   Confidence: {single_result['confidence']:.3f}")
            print(f"   Malignant Probability: {single_result['malignant_probability']:.3f}")
    
    print(f"\n✅ Inference demo completed!")
    print(f"📁 Results saved to: {results_dir}/")
    print(f"\nFiles generated:")
    print(f"  - prediction_report.txt: Comprehensive performance analysis")
    print(f"  - confident_predictions.png: Most confident model predictions")
    print(f"  - random_predictions.png: Random sample predictions")


if __name__ == "__main__":
    """Run the prediction demonstration."""
    main_demo()
