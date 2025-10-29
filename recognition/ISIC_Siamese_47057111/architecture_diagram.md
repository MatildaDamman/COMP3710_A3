# Siamese Network Architecture (final_attempt.py)

Below is a diagram describing the architecture used in `final_attempt.py`:

```
+-------------------+      +-------------------+      +-------------------+
|   Image 1 (224x224) |    |   Image 2 (224x224) |    |   ...             |
+-------------------+      +-------------------+      +-------------------+
         |                          |                          |
         v                          v                          v
+-------------------+      +-------------------+      +-------------------+
|  ResNet18 Backbone |      |  ResNet18 Backbone |      |  ...             |
|  (pretrained, FC removed) |  (shared weights) |      |                  |
+-------------------+      +-------------------+      +-------------------+
         |                          |                          |
         v                          v                          v
+-------------------+      +-------------------+      +-------------------+
| Embedding Network  |      | Embedding Network  |      |                  |
| (512 -> 256 -> 128) |    | (shared weights)   |      |                  |
+-------------------+      +-------------------+      +-------------------+
         |                          |                          |
         +------------+-------------+                          |
                      |                                        |
                      v                                        |
           +-----------------------------+
           |   Concatenate Embeddings    |
           |   (128 + 128 = 256)        |
           +-----------------------------+
                      |
                      v
           +-----------------------------+
           |   Comparison Network        |
           |   (256 -> 512 -> 128 -> 1) |
           +-----------------------------+
                      |
                      v
           +-----------------------------+
           |   Output: Similarity Score  |
           |   (sigmoid for binary)      |
           +-----------------------------+
```

**Legend:**
- ResNet18 Backbone: Feature extractor, shared weights for both images
- Embedding Network: Reduces features to 128-dim vector
- Comparison Network: Fully connected layers to compare embeddings and output similarity
- Output: Probability that the pair is of the same class (benign/malignant)

---

## How to Interpret the Diagram
- Each image is processed independently through the same backbone and embedding network.
- Embeddings are concatenated and passed through the comparison network.
- The final output is a similarity score (used for classification).

---

## Visualisation File
This diagram is provided in text form for clarity. For a graphical version, you can use tools like [Netron](https://netron.app/) or [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) to visualize the PyTorch model.
