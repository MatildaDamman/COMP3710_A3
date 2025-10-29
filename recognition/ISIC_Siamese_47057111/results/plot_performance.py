import re
import matplotlib.pyplot as plt

epochs = []
train_acc = []
val_acc = []
train_loss = []
val_loss = []

with open('final_attempt_318336.log') as f:
    for line in f:
        m = re.search(r'Epoch (\d+)/\d+', line)
        if m:
            epoch = int(m.group(1))
        m2 = re.search(r'Epoch Summary: Train Loss=([0-9.]+), Train Acc=([0-9.]+)%, Val Loss=([0-9.]+), Val Acc=([0-9.]+)%,', line)
        if m2:
            train_loss.append(float(m2.group(1)))
            train_acc.append(float(m2.group(2)))
            val_loss.append(float(m2.group(3)))
            val_acc.append(float(m2.group(4)))
            epochs.append(len(epochs)+1)

plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('performance_accuracy.png')

plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('performance_loss.png')
