import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import warnings
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from torchsummary import summary
from torchvision.models import ResNet34_Weights
from torchvision.ops import sigmoid_focal_loss
from PIL import Image, ImageDraw

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)

def conv4x4(in_planes: int, out_planes: int, stride: int = 2, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = None
        
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )

    def forward(self, x: Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_res_block(blocks, in_dim, out_dim, stride=1):
    layers = []
    layers.append(ResBlock(in_dim, out_dim, stride))
    for _ in range(1, blocks):
        layers.append(ResBlock(out_dim, out_dim))
    return nn.Sequential(*layers)

class Concat(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = conv1x1(in_dim*2, out_dim)
        self.upsample = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x1, x2):
        x2 = self.upsample(x2)
        return self.conv(torch.cat([x1, x2], 1)) # F.interpolate(x2, scale_factor=2, mode='nearest')

class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, k=5):
        super().__init__()
        h_dim = in_dim // 2
        self.cv1 = conv1x1(in_dim, h_dim)
        self.cv2 = conv1x1(h_dim*4, out_dim)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
   
class Encoder(nn.Module):
    """
    maps to the latent space x -> z.
    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """
    def __init__(self, in_dim, h_dim, n_res_layers, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            conv4x4(in_dim, h_dim, 2), # 320x320x16
            nn.ReLU(),
            conv4x4(h_dim, h_dim*2, 2), # 160x160x32
            nn.ReLU()
            )
        self.conv2 = make_res_block(n_res_layers[0], h_dim*2, h_dim*2, 1) # 160x160x128
        self.conv3 = make_res_block(n_res_layers[1], h_dim*2, h_dim*4, 2) # 80x80x256
        self.conv4 = make_res_block(n_res_layers[2], h_dim*4, h_dim*8, 2) # 40x40x512
        self.conv5 = make_res_block(n_res_layers[3], h_dim*8, h_dim*8, 2) # 20x20x512
        self.sppf = SPPF(h_dim*8, h_dim*8, 5)
        self.cat6 = Concat(h_dim*8, h_dim*8)
        self.conv6 = make_res_block(2, h_dim*8, h_dim*4, 1)
        self.cat7 = Concat(h_dim*4, h_dim*4)
        self.conv7 = make_res_block(2, h_dim*4, embedding_dim, 1)
        # self.cat8 = Concat(h_dim*2, h_dim*2)
        # self.conv8 = make_res_block(2, h_dim*2, embedding_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        
        p2 = self.conv2(x)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)
        p5 = self.sppf(p5)
        
        p6 = self.cat6(p4, p5)
        p6 = self.conv6(p6)
        p7 = self.cat7(p3, p6)
        p7 = self.conv7(p7)
        # p8 = self.cat8(p2, p7)
        # p8 = self.conv8(p8)
        return p5, p6, p7
    
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
    def __init__(self, out_dim, h_dim, n_res_layers, embedding_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            conv1x1(embedding_dim, h_dim*4),
            make_res_block(n_res_layers, h_dim*4, h_dim*4, 1),
            nn.ConvTranspose2d(h_dim*4, h_dim*2, 
                               kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(h_dim*2, h_dim, 
                               kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(h_dim, out_dim, 
                               kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

class Remixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Remixer, self).__init__()
        self.in_conv = nn.Sequential(
            conv1x1(in_dim, out_dim),
            nn.ReLU(),
            conv1x1(out_dim, out_dim),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            conv1x1(out_dim*2, out_dim*2),
            nn.ReLU(),
            conv1x1(out_dim*2, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x, x_decoder):
        x = self.in_conv(x)
        return self.out_conv(torch.cat([x, x_decoder], 1))

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
                 in_dim=3,
                 out_dim=16, 
                 h_dim=32, 
                 num_res_layers=[2, 3, 4, 2, 2], 
                 num_embeddings=512, 
                 embedding_dim=64,
                 decay=None,
                 commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.encoder = Encoder(in_dim, h_dim, num_res_layers, embedding_dim)
        if decay:
            self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # self.decoder = Decoder(in_dim*out_dim, h_dim, num_res_layers[4], embedding_dim)
        self.decoder = Decoder(out_dim, h_dim, num_res_layers[4], embedding_dim)
        self.remixer = Remixer(in_dim, out_dim)

    def forward(self, x):
        _, _, z = self.encoder(x)
        quantized, vq_loss, perplexity, encodings = self.vq(z)
        # if perplexity > 30:
        x_decoder = self.decoder(quantized)
        x_recon = self.remixer(x, x_decoder)
        # else:
        
        # x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1, -1, -1)
        # x_filter = self.decoder(z)
        # B, C, H, W = x_filter.shape
        # x_filter = x_filter.view(B, self.out_dim, self.in_dim, H, W)
        # x_recon = (x_filter*x_expanded).sum(dim=2)
        return x_recon, vq_loss, perplexity, quantized

if __name__ == '__main__':
    # encoder = Encoder(3, 32, (2, 3, 4, 2), 128)
    # decoder = Decoder(16, 32, 2, 128)
    x = torch.rand((2,3,640,640))
    # p5, p6, p7 = encoder(x)
    # out = decoder(p7)
    # print(p5.shape, p6.shape, p7.shape, out.shape)
    model = VQVAE(embedding_dim=128)
    out, _, _, z = model(x)
    print(out.shape, z.shape)
    summary(model, input_size=(3,1024,1024), batch_size=2, device='cpu')