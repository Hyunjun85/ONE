import torch
import torch.nn as nn


# model
class net_LogSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.LogSoftmax()

    def forward(self, input):
        return self.op(input)


_model_ = net_LogSoftmax()

# dummy input for onnx generation
_dummy_ = torch.randn(1, 2, 3, 3)
