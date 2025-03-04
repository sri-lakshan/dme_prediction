import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame.columns = self.data_frame.columns.str.strip()
        self.img_dir = img_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        dr_grade = int(row['diagnosis'])
        dme = int(row['adjudicated_dme'])
        return {
            'image': image,
            'dr_grade': dr_grade,
            'dme': dme
        }
