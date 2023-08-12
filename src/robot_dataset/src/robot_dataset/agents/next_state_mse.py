import torch
import torch.nn as nn

class NextStateMSE:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def update(self, model, batch, data_transform, optimizer, device="cuda"):
        model.train()
        optimizer.zero_grad()
        states, actions, ground_truth = data_transform(batch)
        states = states.to(device)
        actions = actions.to(device)
        ground_truth = ground_truth.to(device)
        next_state_diffs = model(states, actions)
        loss = self.criterion(ground_truth, next_state_diffs)
        loss.backward()
        optimizer.step()

        training_metrics = {
            'loss': loss.item()
        }
        return training_metrics