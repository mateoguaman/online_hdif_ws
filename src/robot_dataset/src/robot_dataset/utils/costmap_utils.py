import numpy as np
import torch
from torchvision import transforms as T

def rosmsgs_to_maps(rgbmap, heightmap):
    '''Converts input rgbmaps and heightmaps from numpy arrays incoming from ros msgs to tensors that can be passed into produce_costmap.

    Args:
        - rgbmap:
            HxWx3 Uint8 array containing rgbmap input from ros topic.
        - heightmap:
            HxWx4 Float array containing the following info about heightmap: min, max, mean, std.
    Returns:
        - maps:
            Dictionary containing two tensors:
            {
                'rgb_map': Tensor(C,H,W) where C=3 corresponding to RGB values,
                'height_map': Tensor(C,H,W) where C=5 corresponding to min,     max, mean, std, invalid_mask where 1's correspond to invalid cells
            }
    '''
    ## First, convert rgbmap to tensor
    img_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    rgb_map_tensor = img_transform(rgbmap.astype(np.uint8))
    # Change axes so that map is aligned with robot-centric coordinates
    rgb_map_tensor = rgb_map_tensor.permute(0,2,1)

    ## Now, convert heightmap to tensor
    hm = torch.from_numpy(heightmap)
    hm_nan = torch.isnan(hm).any(dim=-1, keepdim=True) | (hm > 1e5).any(dim=-1, keepdim=True) | (hm < -1e5).any(dim=-1, keepdim=True)
    hm = torch.nan_to_num(hm, nan=0.0, posinf=2, neginf=-2)
    hm = torch.clamp(hm, min=-2, max=2)
    hm = (hm - (-2))/(2 - (-2))
    hm = torch.cat([hm, hm_nan], dim=-1)
    hm = hm.permute(2,0,1)
    height_map_tensor = hm.permute(0,2,1)

    maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

    return maps