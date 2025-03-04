import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from utils.dataloader import FundusDataset
from models.radiologist import MaxViT_DME

def eval(val_csv, img_dir, model_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FundusDataset(val_csv, img_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MaxViT_DME(num_dr_grade_classes=5, dr_embedding_dim=16, num_dme_classes=2)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    true_dme = []
    pred_dme = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            dr_grades = batch['dr_grade'].to(device).long()
            dme_labels = batch['dme'].to(device)
            
            outputs = model(images, dr_grades)
            _, preds = torch.max(outputs, 1)
            
            true_dme.extend(dme_labels.cpu().numpy())
            pred_dme.extend(preds.cpu().numpy())

    print("DME Classification Report:")
    print(classification_report(true_dme, pred_dme, target_names=['No DME', 'DME']))

    cm = confusion_matrix(true_dme, pred_dme)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No DME', 'DME'], yticklabels=['No DME', 'DME'])
    plt.title("Confusion Matrix - DME")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "main":
    eval("C:\\Users\\Lakshan\\Downloads\\messidor\\data\\val_data.csv", \
         "C:\\Users\\Lakshan\\Downloads\\messidor\\data\\images", \
         "model_weights.pth")