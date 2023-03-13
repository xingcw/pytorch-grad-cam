import torch
import numpy as np

from typing import List, Tuple, Dict
from gradcam.base_cam import BaseCAM
from pipelines.utils.constants import IMG_WIDTH, IMG_HEIGHT


def reshape_transform(tensor, height=14, width=14):
    b, nt, nh = tensor.size()   # batch, num_tokens, num_hidden
    result = tensor[:, :-1 , :].reshape(b, height, width, nh)

    # Bring the channels to the first dimension, like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result


class MultiMAECAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=reshape_transform):
        super(
            MultiMAECAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
    
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.nn.Module],
                eigen_smooth: bool = False, 
                **kwargs) -> np.ndarray:
        
        if self.compute_input_gradient:
            for domain, input in inputs.items():
                inputs[domain] = torch.autograd.Variable(input, requires_grad=True)

        preds, masks = self.activations_and_grads(inputs, **kwargs)
        
        for domain in ["rgb", "depth", "semseg"]:
            if domain not in masks:
                device = "cuda" if self.cuda else "cpu"
                masks[domain] = torch.ones_like(list(masks.values())[0]).to(device)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(preds[domain], torch.zeros_like(inputs[domain]), masks[domain]) if domain != "semseg" 
                        else target(preds[domain]) for domain, target in targets.items()])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(inputs,
                                                   targets,
                                                   eigen_smooth)
        cam = self.aggregate_multi_layers(cam_per_layer)
        return preds, masks, cam
    
    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        return IMG_WIDTH, IMG_HEIGHT