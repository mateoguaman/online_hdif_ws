import torch
import torch.nn as nn

def to_device(data, device="cuda"):
    if isinstance(data, dict):
        data = {k:to_device(v, device) for k,v in data.items()}
    else:
        data = data.to(device) if hasattr(data, "to") else data

    return data 

class MSE:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def update(self, model, batch, data_transform, optimizer, device="cuda"):
        # import pdb;pdb.set_trace()
        model.train()
        optimizer.zero_grad()
        input_data, ground_truth = data_transform(batch)
        input_data = to_device(input_data, device)
        ground_truth = to_device(ground_truth, device).double()
        output = model(input_data).double()
        loss = self.criterion(output, ground_truth)
        loss.backward()
        optimizer.step()

        training_metrics = {
            'loss': loss.item()
        }
        return training_metrics