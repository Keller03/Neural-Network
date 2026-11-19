import numpy as np

from src_to_implement.Layers.Base import BaseLayer

# TODO: Implement class CrossEntropyLoss
class CrossEntropyLoss(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_tensor, target_tensor):
        self.predicted_tensor = predicted_tensor
        self.target_tensor = target_tensor

        loss = -np.sum(target_tensor * np.log(predicted_tensor + np.finfo(float).eps))
        return loss

    def backward(self, label_tensor):
        return -label_tensor / (self.predicted_tensor + np.finfo(float).eps)