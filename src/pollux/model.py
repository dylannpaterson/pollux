import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DenseGridModel(nn.Module):
    def __init__(self, K=3, shape_size=9):
        super(DenseGridModel, self).__init__()
        self.K = K
        self.S2 = shape_size * shape_size
        self.num_output_channels = self.K * (5 + self.S2) + 1

        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.layer1, 
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_output_channels, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        
        B, C, H, W = out.shape
        star_out = out[:, :-1, :, :]
        bg_out = out[:, -1:, :, :]
        
        star_out = star_out.view(B, self.K, 5 + self.S2, H, W)
        star_out = star_out.permute(0, 3, 4, 1, 2)
        
        p = torch.sigmoid(star_out[..., 0:1])
        dx = torch.sigmoid(star_out[..., 1:2]) * 2.0
        dy = torch.sigmoid(star_out[..., 2:3]) * 2.0
        m = star_out[..., 3:4]
        c = torch.sigmoid(star_out[..., 4:5])
        
        shape_logits = star_out[..., 5:]
        shape = F.softmax(shape_logits, dim=-1)
        
        bg = F.relu(bg_out.permute(0, 2, 3, 1))
        
        return {
            "stars": torch.cat([p, dx, dy, m, c, shape], dim=-1),
            "background": bg
        }
