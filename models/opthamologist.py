import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataloader import FundusDataset
from radiologist import MaxViT_DME

def train_model(train_csv, img_dir, num_epochs=10, batch_size=16, learning_rate=1e-4, save_path="../model_weights.pth"):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FundusDataset(train_csv, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MaxViT_DME(num_dr_grade_classes=5, dr_embedding_dim=16, num_dme_classes=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            dr_grades = batch['dr_grade'].to(device).long() 
            dme_labels = batch['dme'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, dr_grades)
            loss = criterion(outputs, dme_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
