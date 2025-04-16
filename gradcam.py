import numpy as np
import cv2
import torch
import torch.nn.functional as F
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, target_class, require_output=False):
        self.model.zero_grad()
        output = self.model(x)
        loss = output[:, target_class].sum()
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3))
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= weights[:, i].unsqueeze(-1).unsqueeze(-1)
        # activations *= gradients

        cam = torch.mean(activations, dim=1)

        cam = F.relu(cam)
        cam_max = torch.max(cam, dim=1, keepdim=True)[0]
        cam_max = torch.max(cam_max, dim=2, keepdim=True)[0]
        cam = cam / (cam_max + 1e-20)
        interpolated_cam = F.interpolate(cam.unsqueeze(1), size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        if require_output:
            return interpolated_cam.squeeze(1), output
        else:
            return interpolated_cam.squeeze(1)