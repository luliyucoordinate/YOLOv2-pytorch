import torch

class ReorgLayer(torch.nn.Module):
    def __init__(self, stride):
        super(ReorgLayer, self).__init__()

        self.stride = stride

    def forward(self, input):
        stride = self.stride
        B,C,H,W = input.size()
        ws = stride
        hs = stride
        input = input.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        input = input.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        input = input.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        input = input.view(B, hs*ws*C, H//hs, W//ws)
        return input
