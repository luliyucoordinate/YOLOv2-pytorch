import torch

class RouteLayer(torch.nn.Module):
    def __init__(self):
        super(RouteLayer, self).__init__()

    def forward(self, input):
        return input