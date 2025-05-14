import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# ======== CONFIG ========
CSV_PATH = "data/emotions.csv"  # <-- change to your actual CSV path
IMAGE_FOLDER = "data/images"    # <-- change to your actual image folder
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "emotion_model.pth"
# =========================

# Load CSV
df = pd.read_csv(CSV_PATH, header=None, names=["filename", "emotion", "ignore"])
df["filepath"] = df["filename"].apply(lambda x: os.path.join(IMAGE_FOLDER, x))

# Label encoding
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["emotion"])

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset and DataLoader
dataset = EmotionDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
num_classes = len(label_encoder.classes_)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder.classes_.tolist()
}, MODEL_SAVE_PATH)

print(f"\nModel saved to {MODEL_SAVE_PATH}")

# ======= PREDICTION FUNCTION ========
def predict_emotion(image_path, model_path=MODEL_SAVE_PATH):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    classes = checkpoint['label_encoder']
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()

    return dict(zip(classes, probs))

# Example usage:
# result = predict_emotion("some_new_image.jpg")
# print(result)
