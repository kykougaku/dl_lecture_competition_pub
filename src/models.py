import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models import create_model

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

def tanhexp(x):
    return x * torch.tanh(torch.exp(x))

class myclassifier(nn.Module):
    def __init__(self, in_class, dir) -> None:
        super().__init__()

        self.conv = create_model("efficientnet_b0", num_classes=in_class, in_chans=1, pretrained=True)
        self.conv.load_state_dict(torch.load(dir))
        self.conv.classifier = nn.Identity()

        for param in self.conv.parameters():
            param.requires_grad = False

        self.lenear = nn.Linear(in_features=1280+4, out_features=8000, bias=True)
        self.dropout = nn.Dropout(p=0.01)
        self.lenear2 = nn.Linear(in_features=8000, out_features=4000, bias=True)
        self.fc = nn.Linear(in_features=4000, out_features=1854, bias=True)

    def forward(self, X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        baseoutput = self.conv(X)

        classifierinput = torch.cat((baseoutput, idx), dim=1)
        x = self.lenear(classifierinput)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.lenear2(x)
        x = tanhexp(x)
        x = self.fc(x)

        return x

import torch
import torch.nn as nn


class ResNet_1D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):

    def __init__(self, kernels, in_channels=20, fixed_kernel_size=17, num_classes=6):
        super(EEGNet, self).__init__()
        self.kernels = kernels
        self.planes = 240
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.is_reset = False
        
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=1, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=5, padding=2)
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=128, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(in_features=496, out_features=num_classes)
        self.rnn1 = nn.GRU(input_size=156, hidden_size=156, num_layers=1, bidirectional=True)

    def _make_resnet_layer(self, kernel_size, stride, blocks=9, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
            layers.append(ResNet_1D_Block(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsampling=downsampling))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        
        out = out.reshape(out.shape[0], -1)  

        rnn_out, _ = self.rnn(x.permute(0,2, 1))
        new_rnn_h = rnn_out[:, -1, :]  

        new_out = torch.cat([out, new_rnn_h], dim=1)  
        
        if not self.is_reset:
            result = self.fc(new_out)  
        else:
            result = new_out

        return result
    
    def reset_classifier(self):
        self.is_reset = True


class myEEGnet(nn.Module):
    def __init__(self, in_channel: int, kernels: list, classes: int) -> None:
        super().__init__()
    
        self.kernels = kernels
        self.inchannel = in_channel
        self.classes = classes
        self.channel2 = 128
        self.channel3 = 256
        self.channel4 = 512

        self.conv1 = self.__make_multi_conv(inchannel=in_channel, kernels=kernels, outchannel=self.channel2, stride=1)
        self.bn1 = nn.BatchNorm1d(num_features=self.channel2*len(kernels)+in_channel)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = self.__make_multi_conv(inchannel=self.channel2*len(kernels)+in_channel, kernels=kernels, outchannel=self.channel3, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=self.channel3*len(kernels)+self.channel2*len(kernels)+in_channel)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(in_channels=self.channel3*len(kernels)+self.channel2*len(kernels)+in_channel, out_channels=self.channel4, kernel_size=5, stride=2, padding='valid')
        self.bn3 = nn.BatchNorm1d(num_features=self.channel4)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(in_features=8192, out_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=classes)

    def __make_multi_conv(self, inchannel: int, kernels: list, outchannel: int, stride:int) -> nn.ModuleList:
        convs = nn.ModuleList()
        for kernel in kernels:
            conv = nn.Conv1d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, stride=stride, padding='same')
            convs.append(conv)
        return convs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outlist = []
        outlist.append(X)
        for conv in self.conv1:
            outlist.append(conv(X))
        X = torch.cat(outlist, dim=1)
        X = self.bn1(X)
        X = F.silu(X)
        X = self.avgpool(X)

        outlist = []
        outlist.append(X)
        for conv in self.conv2:
            outlist.append(conv(X))
        X = torch.cat(outlist, dim=1)
        X = self.bn2(X)
        X = F.silu(X)
        X = self.avgpool2(X)

        X = self.conv3(X)
        X = self.bn3(X)
        X = F.silu(X)
        X = self.avgpool3(X)

        X = X.view(X.size(0), -1)
        #X = self.dropout(X)

        out = self.fc1(X)
        out = F.silu(out)
        out = self.fc2(out)
        out = F.silu(out)

        return out