#!/usr/bin/env python3
"""
COMP3710 Assignment 3 - Siamese Inference (CLI)

Adds CLI flags and confusion matrix generation for the RefinedSiameseNetwork.

Usage examples:
  - Single image (uses reference gallery from train/val CSVs):
      python predict.py \
        --image archive/train-image/ISIC_0082934.jpg \
        --checkpoint checkpoints/final_attempt_best.pth \
        --output results

  - Batch evaluation on validation split (saves confusion_matrix.png):
      python predict.py \
        --input-dir archive/train-image/ \
        --checkpoint checkpoints/final_attempt_best.pth \
        --output results
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the exact model used during training
from final_attempt import RefinedSiameseNetwork


def device_auto() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = RefinedSiameseNetwork(dropout_rate=0.3).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped dict
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    return VAL_TRANSFORM(img)


def build_reference_embeddings(train_csv: str,
                               val_csv: str,
                               image_dir: str,
                               model: nn.Module,
                               device: torch.device,
                               per_class: int = 10) -> Dict[int, torch.Tensor]:
    """Build a small reference gallery (embeddings) for each class."""
    # Prefer train split if present, otherwise fall back to val split
    df_paths = [p for p in [train_csv, val_csv] if p and os.path.exists(p)]
    if not df_paths:
        raise FileNotFoundError("No split CSVs found; expected train/val CSVs under archive/.")

    df = pd.concat([pd.read_csv(p) for p in df_paths], ignore_index=True)
    if 'target' not in df.columns or 'isic_id' not in df.columns:
        raise ValueError("CSV must contain 'isic_id' and 'target' columns")

    refs = {}
    with torch.no_grad():
        for cls in [0, 1]:  # 0=Benign, 1=Malignant
            candidates = df[df['target'] == cls]
            emb_list: List[torch.Tensor] = []
            for _, row in candidates.iterrows():
                img_path = os.path.join(image_dir, f"{row['isic_id']}.jpg")
                if not os.path.exists(img_path):
                    continue
                img = load_image(img_path).unsqueeze(0).to(device)
                emb = model.forward_single(img)  # [1, 128]
                emb_list.append(emb.squeeze(0).cpu())
                if len(emb_list) >= per_class:
                    break

            if not emb_list:
                raise RuntimeError(f"No reference images found for class {cls} in {image_dir}")

            refs[cls] = torch.stack(emb_list, dim=0)  # [N_ref, 128]

    return refs


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    return (a @ b.T)


def classify_embedding(emb: torch.Tensor,
                       refs: Dict[int, torch.Tensor],
                       topk: int = 3) -> Tuple[int, float]:
    """Classify an embedding by nearest neighbors in reference gallery."""
    sims = {}
    for cls, ref_embs in refs.items():
        s = cosine_sim(emb.unsqueeze(0), ref_embs).squeeze(0)  # [N_ref]
        top_vals, _ = torch.topk(s, k=min(topk, s.numel()))
        sims[cls] = top_vals.mean().item()

    # Softmax over mean similarities to produce confidence-like score
    vals = np.array([sims.get(0, -1e9), sims.get(1, -1e9)], dtype=np.float32)
    probs = np.exp(vals - vals.max())
    probs = probs / probs.sum()
    pred_cls = int(np.argmax(probs))
    confidence = float(probs[pred_cls])
    return pred_cls, confidence


def evaluate_split(val_csv: str,
                   image_dir: str,
                   model: nn.Module,
                   device: torch.device,
                   refs: Dict[int, torch.Tensor],
                   out_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(val_csv)
    y_true: List[int] = []
    y_pred: List[int] = []
    y_conf: List[float] = []

    with torch.no_grad():
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, f"{row['isic_id']}.jpg")
            if not os.path.exists(img_path):
                continue
            img = load_image(img_path).unsqueeze(0).to(device)
            emb = model.forward_single(img).squeeze(0).cpu()
            pred, conf = classify_embedding(emb, refs)
            y_true.append(int(row['target']))
            y_pred.append(pred)
            y_conf.append(conf)

    if not y_true:
        raise RuntimeError("No evaluable samples found for confusion matrix generation.")

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Val)')
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'])
    with open(os.path.join(out_dir, 'prediction_report.txt'), 'w') as f:
        f.write(report)

    # Predictions CSV
    pred_df = pd.DataFrame({
        'isic_id': [str(x) for x in df['isic_id'].tolist()[:len(y_pred)]],
        'target': y_true,
        'pred': y_pred,
        'confidence': y_conf,
    })
    pred_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)

    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description='Siamese inference and evaluation')
    parser.add_argument('--image', type=str, help='Path to a single image (.jpg) to classify')
    parser.add_argument('--input-dir', type=str, help='Directory of images to evaluate (uses val CSV labels)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_attempt_best.pth', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    # Optional (defaults to repo paths if present)
    parser.add_argument('--train_csv', type=str, default='archive/train_split.csv', help='Training split CSV')
    parser.add_argument('--val_csv', type=str, default='archive/val_split.csv', help='Validation split CSV')
    parser.add_argument('--image_dir', type=str, default='archive/train-image/', help='Directory containing ISIC images')
    parser.add_argument('--per_class', type=int, default=10, help='Reference images per class for gallery')
    parser.add_argument('--topk', type=int, default=3, help='k for top-k similarity averaging')

    args = parser.parse_args()

    # Resolve relative paths: defaults relative to script dir, user-specified relative to CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.abspath(p)

    def resolve_default_or_user(path_val: str, default_key: str) -> str:
        default_val = parser.get_default(default_key)
        if path_val == default_val:
            # Default: make relative to script directory
            return default_val if os.path.isabs(default_val) else os.path.join(script_dir, default_val)
        # User-provided: resolve relative to current working directory
        return resolve(path_val)

    args.checkpoint = resolve_default_or_user(args.checkpoint, 'checkpoint')
    args.image_dir = resolve_default_or_user(args.image_dir, 'image_dir')
    args.output = resolve_default_or_user(args.output, 'output')
    args.train_csv = resolve_default_or_user(args.train_csv, 'train_csv') if args.train_csv else None
    args.val_csv = resolve_default_or_user(args.val_csv, 'val_csv') if args.val_csv else None

    # Validate basic paths
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")
    os.makedirs(args.output, exist_ok=True)

    device = device_auto()
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)

    # Build reference gallery
    refs = build_reference_embeddings(
        train_csv=args.train_csv if os.path.exists(args.train_csv) else None,
        val_csv=args.val_csv if os.path.exists(args.val_csv) else None,
        image_dir=args.image_dir,
        model=model,
        device=device,
        per_class=args.per_class,
    )

    # Single image classification
    if args.image:
        img_path = args.image
        if not os.path.exists(img_path):
            # allow relative basename by searching in image_dir
            candidate = os.path.join(args.image_dir, os.path.basename(img_path))
            if os.path.exists(candidate):
                img_path = candidate
            else:
                raise FileNotFoundError(f"Image not found: {args.image}")
        else:
            # If provided as relative path, resolve relative to CWD OR script dir
            img_path = img_path if os.path.isabs(img_path) else resolve(img_path)

        with torch.no_grad():
            img = load_image(img_path).unsqueeze(0).to(device)
            emb = model.forward_single(img).squeeze(0).cpu()
            pred_cls, conf = classify_embedding(emb, refs, topk=args.topk)

        label_name = 'Malignant' if pred_cls == 1 else 'Benign'
        print(f"Image: {img_path}")
        print(f"Prediction: {label_name}")
        print(f"Confidence: {conf:.3f}")

    # Batch evaluation (uses labels from val_csv if present)
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
        if not os.path.exists(args.val_csv):
            print("Warning: --val_csv not found; confusion matrix will be skipped.")
        else:
            print("Running evaluation on validation split and saving confusion_matrix.png …")
            y_true, y_pred = evaluate_split(
                val_csv=args.val_csv,
                image_dir=args.image_dir,
                model=model,
                device=device,
                refs=refs,
                out_dir=args.output,
            )
            acc = (y_true == y_pred).mean()
            print(f"Validation accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    main()
