import torch
import numpy as np

from typing import List, Tuple
from gradcam.base_cam import BaseCAM
from pipelines.utils.constants import IMG_WIDTH, IMG_HEIGHT


class ActionErrTarget:
    def __init__(self, act_label):
        self.label = act_label
    
    def __call__(self, student_act):
        return torch.mean((student_act - self.label) ** 2)


def reshape_transform(tensor, height=14, width=14):
    b, nt, nh = tensor.size()   # batch, num_tokens, num_hidden
    result = tensor[:, :-1 , :].reshape(b, height, width, nh)

    # Bring the channels to the first dimension, like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result


class StudentCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=reshape_transform):
        super(
            StudentCAM,
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
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False, 
                **kwargs) -> np.ndarray:
        
        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.activations_and_grads(input_tensor, **kwargs)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)
            
        input_tensor = input_tensor["rgb"]

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
    
    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        return IMG_WIDTH, IMG_HEIGHT