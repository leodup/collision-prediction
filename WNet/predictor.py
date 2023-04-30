print(".py is running...")
print("Importing libraries...")
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from UNet_Predictor import UNet_Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

print("Loading Data...")
# Import Tensor Data
# Train
masks1 = torch.load('WNet/data/masks.pt')
masks2 = torch.load('WNet/data/unlabeled_masks.pt')
masks = torch.cat((masks1, masks2), dim=0)
# Validation
val_masks = torch.load('WNet/data/val_masks.pt')
print("Masks:", masks.shape)
print("Validation Masks:", val_masks.shape)
print()

print("Loaders...\n")
# Dataset Class & Loaders
class MaskDataset(Dataset):
  def __init__(self, masks, transform=None):
    self.masks = masks.reshape(-1, 22, 160, 240)
    self.t = transforms.RandomHorizontalFlip()
  
  def __len__(self):
    return len(self.masks)

  def __getitem__(self, index):
    msk = self.masks[index]
    msk = self.t(msk) # Random Horizontal Flip: augmentation

    return msk

# Datasets
train_dataset = MaskDataset(masks)
val_dataset = MaskDataset(val_masks)

batch_size = 16 # 64 leads to memory shortage on our systems

# Data Loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


print("Building model...")
model = UNet_Predictor().to(device)
model.load_state_dict(torch.load("WNet/predictor_models/predictor_2.pth")) # LOAD TRAINED MODEL
input_tensor = train_dataset[0].unsqueeze(0)[:, :11, :, :].to(device).long()
output = model(input_tensor)
print("Model output:", output.shape)
print(f"Number of Weights: {sum(p.numel() for p in model.parameters()):,}\n\n")

result = {"train": [], "val": []} # For tracking loss
best_val = -1 # Negative as real val will be positive


optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
criterion = nn.CrossEntropyLoss()

print("Now training:")

# Training
num_epochs = 10

def get_loss(seq, targ, optimizer=None):
  pred_mask = model(seq.long()).squeeze(1)
  loss = criterion(pred_mask, targ.long())

  if optimizer is not None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return loss

for epoch in range(1, num_epochs+1): # Train on several epochs
  total_train_loss = 0
  model.train()
  num = 6
  for batch in train_loader:
    for i in range(num): # Generate next image 10 times | This is where we shift the model inputs
      mask_seq = batch[:, i:11+i, :, :].to(device)
      targ = batch[:, 11+i, :, :].to(device)
      total_train_loss += get_loss(mask_seq, targ, optimizer=optimizer)

  train_loss = total_train_loss.item() / (num*len(train_loader))
  result["train"].append(train_loss)

  # Test model on validation
  with torch.no_grad():
    val_result = 0
    count = 0
    model.eval()
    total_val_loss = 0
    for batch in val_loader:
      for i in range(11): # Generate next image 10 times | This is where we shift the model inputs
        mask_seq_val = batch[:, i:11+i, :, :].to(device)
        targ_val = batch[:, 11+i, :, :].to(device)
        total_val_loss += get_loss(mask_seq_val, targ_val) * mask_seq_val.size(0)
        count += mask_seq_val.size(0)

    val_result = total_val_loss.item() / count
    result["val"].append(val_result)
    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_result:.4f}")

    if (best_val == -1) or (val_result < best_val):
        best_val = val_result
        torch.save(model.state_dict(), "WNet/predictor_models/job_best_predictor.pth")

  scheduler.step(total_val_loss)

print()
