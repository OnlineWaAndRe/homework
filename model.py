import torch
from torch.nn import functional as F
from torch import nn
from torch import optim


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1,  stride=1),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.extra(x) + out

        return F.relu(out)

class RsNet(nn.Module):
    def __init__(self):
        super(RsNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64)
        )
        self.blk1 = ResBlock(64, 64)
        self.blk2 = ResBlock(64, 128)
        self.outlayer = nn.Linear(128 * 112 * 112, 2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x