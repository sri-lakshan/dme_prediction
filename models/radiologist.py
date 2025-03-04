import torch
import torch.nn as nn
import timm

class MaxViT_DME(nn.Module):
    def __init__(self, num_dr_grade_classes=5, dr_embedding_dim=16, num_dme_classes=2):
        super(MaxViT_DME, self).__init__()
        self.backbone = timm.create_model("maxvit_base_tf_224", pretrained=True)
        self.backbone.reset_classifier(0)
        feature_dim = self.backbone.num_features 

        self.dr_embedding = nn.Embedding(num_dr_grade_classes, dr_embedding_dim)
        fused_dim = feature_dim + dr_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_dme_classes)
        )
        
    def forward(self, image, dr_grade):
        img_features = self.backbone(image)
        dr_embedded = self.dr_embedding(dr_grade)
        fused = torch.cat([img_features, dr_embedded], dim=1)
        out = self.fc(fused)
        return out

