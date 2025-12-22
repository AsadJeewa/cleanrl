import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTorso(nn.Module):
    """
    CNN torso similar to Atari style
    Expects input: N x 1 x H x W (grayscale grid)
    Output: flattened feature vector
    """
    def __init__(self, input_shape, output_dim=512):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1) #greyscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.output_dim = output_dim
        
        # Compute size after convs
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2*padding - (kernel_size-1) -1)//stride +1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64

        self.fc = nn.Linear(linear_input_size, output_dim)

    def forward(self, x):
        x = x.float()  # ensure float
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
