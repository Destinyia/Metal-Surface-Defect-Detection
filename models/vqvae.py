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
    
class ChannelMixer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, channel_shift=1, clip_value=1.0, mul=None):
        super(ChannelMixer, self).__init__()
        # 残差层
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        if mul is None:
            self.conv1x1.weight.data = torch.Tensor([0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]).reshape(1,16,1,1)-0.1
        else:
            self.conv1x1.weight.data = mul
        self.conv1x1.bias.data.fill_(0)
        # 非线性表达分支
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1),
        )
        self.cn = ChannleNormalize()
        self.relu = nn.LeakyReLU(0.1)
        self.clip_value = clip_value
        self.out_channels = out_channels
        self.channel_shift = channel_shift

    def forward(self, x):
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
    
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    
class Encoder(nn.Module):
    """
    maps to the latent space x -> z.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, embedding_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers),
            nn.Conv2d(h_dim, embedding_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
        )

    def forward(self, x):
        return self.conv_stack(x)
    
class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, out_dim, h_dim, n_res_layers, res_h_dim, embedding_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_dim, kernel_size=kernel,
                               stride=stride, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # 论文中损失函数的第三项
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # 论文中损失函数的第二项
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach() # 梯度复制
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape # B(256) H(8) W(8) C(64)
        
        # Flatten input BHWC -> BHW, C
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances 计算与embedding space中所有embedding的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # 取最相似的embedding
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # 映射为 one-hot vector
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) # 根据index使用embedding space对应的embedding
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n) 
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw) 
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1)) # 论文中公式(8)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # 计算encoder输出（即inputs）和decoder输入（即quantized）之间的损失
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach() # trick, 将decoder的输入对应的梯度复制，作为encoder的输出对应的梯度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity, encodings

class VQVAE(nn.Module):
    def __init__(self, 
                 channel_in=3,
                 channel_out=16, 
                 channel_h=64, 
                 num_res_layers=2, 
                 res_h_dim=16, 
                 num_embeddings=512, 
                 embedding_dim=64,
                 decay=None,
                 commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channel_in, channel_h, num_res_layers, res_h_dim, embedding_dim)
        if decay:
            self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(channel_out, channel_h, num_res_layers, res_h_dim, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, encodings = self.vq(z)
        # if perplexity > 30:
        # x_recon = self.decoder(quantized)
        # else:
        x_recon = self.decoder(z)
        return x_recon, vq_loss, perplexity, quantized
    
if __name__ == '__main__':
    model = VQVAE()
    x = torch.rand(2, 3, 512, 512)
    x_recon, l, p, e = model(x)
    print(x_recon.shape)
    