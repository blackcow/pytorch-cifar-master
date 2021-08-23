import torch

class Conv()
def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1,
             bias=True, padding_mode='zeros'):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(Conv2d, self).__init__(
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias, padding_mode)


def _conv_forward(self, input, weight):
    if self.padding_mode != 'zeros':
        return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                        weight, self.bias, self.stride,
                        _pair(0), self.dilation, self.groups)
    return F.conv2d(input, weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


def forward(self, input):
    return self._conv_forward(input, self.weight)
H, W, C = 1920, 1080, 3
input = torch.zeros(H, W, C)
torch.nn.Conv2d(C,C,kernel_size=3,stride=1, padding=1)
