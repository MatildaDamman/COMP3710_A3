import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load train split
df = pd.read_csv('../archive/train_split.csv')

# Select 12 images from those available locally
available_images = set(os.listdir('../archive/train-image'))
examples = df[df['isic_id'].isin([img.replace('.jpg','') for img in available_images])].head(12)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, (idx, row) in enumerate(examples.iterrows()):
    img_path = f"../archive/train-image/{row['isic_id']}.jpg"
    img = Image.open(img_path)
    ax = axes[i//4, i%4]
    ax.imshow(img)
    label = 'Melanoma' if row['target'] == 1 else 'Not Melanoma'
    ax.set_title(f"{row['isic_id']}\n{label}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('classification_examples.png')
