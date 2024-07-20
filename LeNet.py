import torch
import torch.nn as nn

# LeNet architecture
# 1 * 32 * 32 Input --> (5*5), s=1, p=0 --> avg pool s=2, p=0 --> (5*5), s=1, p=0 --> avg pool s=2, p=0
# --> Conv 5 * 5 to 120 channels * Linear 120 -> 84 * Linear 10

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.relu = nn.ReLU()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride = 1)
    self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = 1)
    self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5, 5), stride = 1)
    self.avg_pool = nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2))
    self.fc1 = nn.Linear(120, 84)
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.relu(self.conv1(x)) # (B, 6, 28, 28)
    x = self.avg_pool(x) # (B, 6, 14, 14)
    x = self.relu(self.conv2(x)) # (B, 16, 10, 10)
    x = self.avg_pool(x) # (B, 16, 5, 5)
    x = self.relu(self.conv3(x)) # (B, 120, 1, 1)
    x = x.reshape(x.shape[0], -1) # (B, 120)
    x = self.relu(self.fc1(x)) # (B, 84)
    x = self.fc2(x) # (B, 10)
    return x

x = torch.rand(64, 1, 32, 32)
le_net = LeNet()
