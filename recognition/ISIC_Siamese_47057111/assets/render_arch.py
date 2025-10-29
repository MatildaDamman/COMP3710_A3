#!/usr/bin/env python3
"""
Quick static renderer: draws a simple Siamese architecture diagram to PNG
so README can embed it without Mermaid.
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 1400, 600
BG = (255, 255, 255)
BOX = (230, 242, 255)
EMB = (255, 244, 225)
CMP = (232, 245, 233)
OUT = (252, 228, 236)
FG = (40, 40, 40)
AR = (100, 100, 100)

img = Image.new('RGB', (W, H), BG)
draw = ImageDraw.Draw(img)
try:
    # Try a common font; fallback to default
    font = ImageFont.truetype("Arial.ttf", 18)
    title_font = ImageFont.truetype("Arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()
    title_font = font

# Helper to draw centered text
def centered_text(xy, text, f=font):
    w, h = draw.textsize(text, font=f)
    draw.text((xy[0]-w/2, xy[1]-h/2), text, fill=FG, font=f)

# Helper to draw box with label

def box(x, y, w, h, fill, text_lines):
    draw.rectangle([x, y, x+w, y+h], fill=fill, outline=FG)
    ty = y + 10
    for i, line in enumerate(text_lines):
        centered_text((x + w/2, ty + i*22), line)

# Helper to draw arrow

def arrow(x1, y1, x2, y2):
    draw.line([x1, y1, x2, y2], fill=AR, width=3)
    # simple arrow head
    ah = 10
    draw.polygon([(x2, y2), (x2-ah, y2-ah), (x2-ah, y2+ah)], fill=AR)

# Layout coordinates
col = [80, 320, 560, 800, 1040, 1220]
y1 = 160

# Input images
box(col[0]-60, y1-40, 120, 80, (245,245,245), ["Image 1", "224×224"])
box(col[0]-60, y1+140, 120, 80, (245,245,245), ["Image 2", "224×224"])

# Backbones (shared)
box(col[1]-110, y1-60, 220, 120, BOX, ["ResNet18 Backbone", "(shared, FC removed)"])
box(col[1]-110, y1+120, 220, 120, BOX, ["ResNet18 Backbone", "(shared, FC removed)"])

# Embeddings
box(col[2]-110, y1-60, 220, 120, EMB, ["Embedding Network", "512→256→128"])
box(col[2]-110, y1+120, 220, 120, EMB, ["Embedding Network", "512→256→128"])

# Concatenate
box(col[3]-110, y1+30, 220, 80, (245,245,255), ["Concatenate", "[emb1; emb2] (256-d)"])

# Comparison network
box(col[4]-130, y1+10, 260, 120, CMP, ["Comparison Network", "256→512→128→1"])

# Output
box(col[5]-80, y1+30, 160, 80, OUT, ["Output", "Similarity score (σ)"])

# Arrows from inputs to backbones
arrow(col[0]+60, y1, col[1]-110, y1)
arrow(col[0]+60, y1+180, col[1]-110, y1+180)

# Arrows from backbones to embeddings
arrow(col[1]+110, y1, col[2]-110, y1)
arrow(col[1]+110, y1+180, col[2]-110, y1+180)

# Arrows from embeddings to concat
arrow(col[2]+110, y1, col[3]-110, y1+70)
arrow(col[2]+110, y1+180, col[3]-110, y1+70)

# Concat to comparison to output
arrow(col[3]+110, y1+70, col[4]-130, y1+70)
arrow(col[4]+130, y1+70, col[5]-80, y1+70)

# Title
centered_text((W/2, 40), "Siamese Network Architecture", title_font)

out_path = os.path.join(os.path.dirname(__file__), 'architecture_diagram.png')
img.save(out_path)
print(f"Saved diagram to {out_path}")
