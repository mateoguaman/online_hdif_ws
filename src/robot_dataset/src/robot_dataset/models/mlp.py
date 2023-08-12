import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, state_dim=3, action_dim=2, latent_dim=32):
        super().__init__()
        self.state_model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
        )

        self.action_model = nn.Sequential(
            nn.Linear(in_features=action_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim)
        )

        self.cat_model = nn.Sequential(
            nn.Linear(in_features=2*latent_dim, out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim, out_features=state_dim)
        )

    def forward(self, state, action):
        state_out = self.state_model(state)
        action_out = self.action_model(action)
        combined = torch.concatenate([state_out, action_out], dim=-1)
        out = self.cat_model(combined)
        return out