import torch
import kornia
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim, MS_SSIM
from torchvision.ops import sigmoid_focal_loss

class RandomHueTransform:
    def __init__(self, hue_range=(-0.1, 0.1), prop=0.2):
        self.hue_range = hue_range
        self.prop = prop

    def __call__(self, x):
        # x: 输入Tensor，假设是 [B, C, H, W] 格式，且数据范围是[0, 1]
        # 创建与x形状相同的张量，并填充为prop
        prob_tensor = torch.full((x.size(0),), fill_value=self.prop, device=x.device)
        # 生成0和1的张量
        bernoulli_tensor = torch.bernoulli(prob_tensor)
        # 生成随机色调值
        hue = torch.empty(x.size(0), device=x.device).uniform_(*self.hue_range) * bernoulli_tensor
        # 应用色调变换
        return kornia.enhance.adjust_hue(x, hue)

class ChannleNormalize(nn.Module):
    def __init__(self):
        super(ChannleNormalize, self).__init__()

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, channels, height, width)
        min_vals = x.view(x.size(0), x.size(1), -1).min(dim=2, keepdim=True)[0].view(x.size(0), x.size(1), 1, 1)
        max_vals = x.view(x.size(0), x.size(1), -1).max(dim=2, keepdim=True)[0].view(x.size(0), x.size(1), 1, 1)
        # 防止除以0
        diff = max_vals - min_vals + 1e-8
        # diff[diff == 0] = 1
        # 标准化到0-1之间
        x = (x - min_vals) / diff
        return x
    
class TaylorSeriesChannelMixer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, darkness=0.0, gamma=1.0, channel_shift=1, clip_value=1.0, mul=None):
        super(TaylorSeriesChannelMixer, self).__init__()
        # 参数初始化
        self.coeff = nn.Parameter(torch.tensor([0,1,0,0,0], dtype=torch.float32))
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        if mul is None:
            self.conv1x1.weight.data = torch.Tensor([0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]).reshape(1,16,1,1)-0.1
        else:
            self.conv1x1.weight.data = mul
        self.conv1x1.bias.data.fill_(0)
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1),
        )
        self.cn = ChannleNormalize()
        self.relu = nn.LeakyReLU(0.1)
        self.clip_value = clip_value
        self.out_channels = out_channels
        self.channel_shift = channel_shift

    def forward(self, x):
        # 应用泰勒展开
        x = self.coeff[0] + \
            self.coeff[1] * x + \
            self.coeff[2] * (x**2) / 2 + \
            self.coeff[3] * (x**3) / 6 + \
            self.coeff[4] * (x**4) / 24
        # x = x + torch.clamp(self.darkness, -0.2, 0.2)
        # gamma = torch.relu(self.gamma) + 0.1
        # x = torch.pow(x, gamma)
        
        x = [self._roll_tensor(x, i*self.channel_shift) for i in range(self.out_channels)]
        x = torch.cat([self.relu(self.conv1x1(x[i]) + self.res_block(x[i])) for i in range(self.out_channels)], dim=1).contiguous()
        x = self.cn(x)
        # x = x.clamp(max=self.clip_value)  # 将输出限制在[0, self.clip_value]
        return x
    
    def _roll_tensor(self, tensor, shift=1):
        if shift == 0:
            return tensor
        # 循环平移操作
        return torch.cat((tensor[:, shift:, :, :], tensor[:, :shift, :, :]), dim=1).contiguous()
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

class ChannelCompression(nn.Module):
    def __init__(self, in_channels, out_channels, clip_value=1.0, mul=None):
        super(ChannelCompression, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        # 初始化权重
        if mul is None:
            self.conv.weight.data = torch.Tensor([0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]).reshape(1,16,1,1)-0.01
        else:
            self.conv.weight.data = mul
        self.conv.bias.data.fill_(0)
        # 截断输出
        self.cn = ChannleNormalize()
        self.clip_value = clip_value
        self.out_channels = out_channels

    def forward(self, x):
        # x = self.conv(x)
        x = torch.cat([self.conv(self._roll_tensor(x, i)) for i in range(self.out_channels)], dim=1).contiguous()
        x = F.leaky_relu(x, 0.1)  # 应用ReLU激活函数
        x = self.cn(x)
        # x = x.clamp(max=self.clip_value)  # 将输出限制在[0, self.clip_value]
        return x
    
    def _roll_tensor(self, tensor, shift=1):
        if shift == 0:
            return tensor
        # 循环平移操作
        return torch.cat((tensor[:, shift:, :, :], tensor[:, :shift, :, :]), dim=1).contiguous()
    
    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

  

if __name__ == '__main__':
    # encoder = Encoder(3, 32, (2, 3, 4, 2), 128)
    # decoder = Decoder(16, 32, 2, 128)
    # p5, p6, p7 = encoder(x)
    # out = decoder(p7)
    # print(p5.shape, p6.shape, p7.shape, out.shape)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.rand((2, 16, 640, 640)).to(device)
    mul = torch.Tensor([[0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]]).reshape(1,16,1,1)-0.1
    channel_compressor = TaylorSeriesChannelMixer(in_channels=16, 
                                                hidden_channels=8, 
                                                out_channels=3, 
                                                darkness=0.0,
                                                gamma=1.0,
                                                channel_shift=1,
                                                clip_value=1.0,
                                                mul=mul).to(device)
    out = channel_compressor(x)
    print(out.shape)
    print(channel_compressor.coeff.cpu().detach().numpy())
    for param in channel_compressor.state_dict().keys():
        print(param)
    # summary(model, input_size=(3,640,640), batch_size=2, device='cpu')