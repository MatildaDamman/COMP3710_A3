#!/usr/bin/env python3
"""
Final Attempt: EXACT 76.25% Model + Optimal Learning Rate
Only change: Learning rate schedule for those final 3.75%
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import random
from typing import Optional, List, Dict, Tuple
import time
from datetime import datetime

def log_with_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()

class WeightedFocalLoss(nn.Module):
    """EXACT SAME as 76.25% version"""
    def __init__(self, alpha=0.7, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class RefinedSiameseNetwork(nn.Module):
    """EXACT SAME as 76.25% success - ZERO changes"""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        log_with_time("🏗️ Building EXACT 76.25% Model - Final Attempt")
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.embedding_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.comparison_net = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(128, 1)
        )
        
        log_with_time("✅ EXACT 76.25% Architecture Ready")
    
    def forward_single(self, x):
        features = self.backbone(x)
        embedding = self.embedding_net(features)
        return embedding
    
    def forward(self, x1, x2):
        emb1 = self.forward_single(x1)
        emb2 = self.forward_single(x2)
        combined = torch.cat([emb1, emb2], dim=1)
        similarity = self.comparison_net(combined)
        return similarity, emb1, emb2

class StrategicDataset(Dataset):
    """EXACT SAME as 76.25% success"""
    def __init__(self, csv_path: str, image_dir: str, transform=None, 
                 max_pairs=250, positive_ratio=0.35):
        
        log_with_time(f"📊 Creating EXACT Same Strategic Dataset")
        
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        self.benign_samples = self.df[self.df['target'] == 0].reset_index(drop=True)
        self.malignant_samples = self.df[self.df['target'] == 1].reset_index(drop=True)
        
        log_with_time(f"   Benign: {len(self.benign_samples)}, Malignant: {len(self.malignant_samples)}")
        
        self.pairs = self._create_strategic_pairs(max_pairs, positive_ratio)
        log_with_time(f"   Created {len(self.pairs)} strategic pairs")
    
    def _create_strategic_pairs(self, max_pairs, positive_ratio):
        pairs = []
        
        n_positive = int(max_pairs * positive_ratio)
        n_negative = max_pairs - n_positive
        
        malignant_positive = min(n_positive // 3, len(self.malignant_samples) // 2)
        for _ in range(malignant_positive):
            if len(self.malignant_samples) >= 2:
                idx1, idx2 = random.sample(range(len(self.malignant_samples)), 2)
                pairs.append({
                    'img1': self.malignant_samples.iloc[idx1],
                    'img2': self.malignant_samples.iloc[idx2],
                    'label': 1
                })
        
        benign_positive = n_positive - malignant_positive
        for _ in range(benign_positive):
            if len(self.benign_samples) >= 2:
                idx1, idx2 = random.sample(range(len(self.benign_samples)), 2)
                pairs.append({
                    'img1': self.benign_samples.iloc[idx1],
                    'img2': self.benign_samples.iloc[idx2],
                    'label': 1
                })
        
        for _ in range(n_negative):
            benign_idx = random.randint(0, len(self.benign_samples) - 1)
            malignant_idx = random.randint(0, len(self.malignant_samples) - 1)
            pairs.append({
                'img1': self.benign_samples.iloc[benign_idx],
                'img2': self.malignant_samples.iloc[malignant_idx],
                'label': 0
            })
        
        random.shuffle(pairs)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        img1_isic_id = str(pair['img1']['isic_id'])
        img2_isic_id = str(pair['img2']['isic_id'])
        
        img1_path = os.path.join(self.image_dir, f"{img1_isic_id}.jpg")
        img2_path = os.path.join(self.image_dir, f"{img2_isic_id}.jpg")
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except:
            img1 = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img2 = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.tensor(pair['label'], dtype=torch.float32)
        return img1, img2, label

class FinalAttemptTrainer:
    """ONLY CHANGE: Optimal learning rate schedule"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # EXACT SAME loss
        self.criterion = WeightedFocalLoss(alpha=0.7, gamma=1.5)
        
        # ONLY CHANGE: Slightly lower starting LR for fine-tuning
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=0.0008,  # Was 0.001 - slightly lower for fine convergence
            weight_decay=0.0001
        )
        
        # ONLY CHANGE: Different scheduler for final push
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor validation accuracy
            factor=0.5,  # Halve LR when plateau
            patience=5,  # Wait 5 epochs before reducing
            verbose=True
        )
        
        self.best_accuracy = 0.0
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (img1, img2, target) in enumerate(dataloader):
            batch_start = time.time()
            
            img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            similarity_pred, emb1, emb2 = self.model(img1, img2)
            loss = self.criterion(similarity_pred.squeeze(), target)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(similarity_pred.squeeze()) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                acc = 100.0 * correct / total
                lr = self.optimizer.param_groups[0]['lr']
                batch_time = time.time() - batch_start
                log_with_time(f"   Batch {batch_idx + 1}: Loss={loss.item():.4f}, Acc={acc:.1f}%, LR={lr:.6f}, Time={batch_time:.2f}s")
        
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, target in dataloader:
                img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
                
                similarity_pred, _, _ = self.model(img1, img2)
                loss = self.criterion(similarity_pred.squeeze(), target)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(similarity_pred.squeeze()) > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        avg_loss = val_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=35):
        log_with_time("🚀 FINAL ATTEMPT - 76.25% + Optimal LR Schedule")
        log_with_time("=" * 60)
        
        for epoch in range(epochs):
            log_with_time(f"\nEpoch {epoch + 1}/{epochs}")
            log_with_time("-" * 40)
            
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            # Step scheduler with validation accuracy
            self.scheduler.step(val_acc)
            
            log_with_time(f"   Epoch Summary: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%, Time={epoch_time:.1f}s")
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save(self.model.state_dict(), 'final_attempt_best.pth')
                log_with_time(f"🎯 NEW BEST: {val_acc:.2f}% validation accuracy!")
                
                if val_acc >= 80.0:
                    log_with_time(f"🏆 TARGET ACHIEVED: {val_acc:.2f}% >= 80%!")
                    break
        
        log_with_time(f"\n✅ Final attempt completed! Best accuracy: {self.best_accuracy:.2f}%")

def main():
    log_with_time("🎯 FINAL ATTEMPT - 76.25% Model + Optimal Learning Rate")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_with_time(f"Device: {device}")
    
    # EXACT SAME transforms as 76.25%
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # EXACT SAME datasets
    train_dataset = StrategicDataset(
        csv_path='archive/train_split.csv',
        image_dir='archive/train-image/image/',
        transform=train_transform,
        max_pairs=250,
        positive_ratio=0.35
    )
    
    val_dataset = StrategicDataset(
        csv_path='archive/val_split.csv',
        image_dir='archive/train-image/image/',
        transform=val_transform,
        max_pairs=100,
        positive_ratio=0.35
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # EXACT SAME model
    model = RefinedSiameseNetwork(dropout_rate=0.3).to(device)
    
    # ONLY CHANGE: Different learning rate schedule
    trainer = FinalAttemptTrainer(model, device)
    
    trainer.train(train_loader, val_loader, epochs=35)

if __name__ == '__main__':
    main()