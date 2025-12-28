# ----------------------------- Imports -----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from tqdm.notebook import tqdm
import os

# ----------------------------- Paths -----------------------------
dataset_path = "/content/drive/MyDrive/Tooth_dataset"  # change if needed
save_path = "/content/dental_model.pth"

# ----------------------------- Transforms -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------- Dataset -----------------------------
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

class TransformedSubset(Subset):
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform:
            x = self.transform(x)
        return x, y

val_dataset = TransformedSubset(val_dataset, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------------------- Device -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------- Model -----------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers except layer4 and fc
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # 5 classes
model = model.to(device)

# ----------------------------- Loss + Optimizer -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ----------------------------- Classes -----------------------------
classes = ['Calculus', 'Caries', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

# ----------------------------- Training Loop -----------------------------
num_epochs = 15
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print batch loss every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)

    # ----------------- Validation -----------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

    # Save checkpoint if validation improves
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), save_path)
        print(f"Validation improved. Model saved at epoch {epoch+1} with Val Accuracy: {best_val_acc:.2f}%")

    scheduler.step()

print(f"Training complete. Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Best model saved at: {save_path}")
