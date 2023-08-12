import torch
import torch.nn as nn
from torchvision import models

class CostFourierVelModel(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        # import pdb;pdb.set_trace()
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        if len(processed_maps.shape) < 2:
            processed_maps = processed_maps.view(1, -1)
        if len(processed_vel.shape) < 2:
            processed_vel = processed_vel.view(1, -1)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output