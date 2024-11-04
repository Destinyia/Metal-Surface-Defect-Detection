import torch
import os, shutil
import numpy as np
import kornia
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim, MS_SSIM
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.fpn_vae import VQVAE
from models.vqvae import RandomHueTransform
from models.mixer import TaylorSeriesChannelMixer
from models.backbones import YOLOv8Backbone
from models.transforms import EllipseNoise, ChannelReduce
from McDataset import McDataset
from util import tensor_to_images, save_images, save_mix_image, save_weights, save_model, load_model_weights

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.set_printoptions(precision=5)

transform = transforms.Compose([
    # transforms.Resize(1280),
    transforms.RandomResizedCrop(size=1024, scale=(0.9, 1.11), ratio=(0.9, 1.11)),  # 随机裁剪图像
    transforms.RandomRotation(degrees=15),  # 随机旋转图像±15度
])
noise_transform = EllipseNoise(num_ellipses=10, dark=0.9, noise_level=0.6)
hue_transform = RandomHueTransform((-0.05, 0.05), 0.5)
reduce_transform = ChannelReduce(reduce_range=(0.3, 0.8))

exp_path = 'runs/pifm/exp17'
if os.path.exists(exp_path):
    shutil.rmtree(exp_path)
os.mkdir(exp_path)

# datasets
train_set = McDataset(root_dir='./datasets/VAE/images', transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_set = McDataset(root_dir='./datasets/VAE/test', transform=transform)
test_loader = DataLoader(test_set, batch_size=2, shuffle=True)
test_x = torch.unsqueeze(test_set[0], dim=0).to(device)

mul = torch.Tensor([[0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0]]).reshape(1,16,1,1)-0.1
# channel_compressor = ChannelCompression(16, 3, 1.0, None).to(device)
channel_compressor = TaylorSeriesChannelMixer(in_channels=16, 
                                                hidden_channels=8, 
                                                out_channels=3, 
                                                darkness=0.0,
                                                gamma=1.0,
                                                channel_shift=1,
                                                clip_value=1.0,
                                                mul=mul).to(device)

feature_extractor = YOLOv8Backbone(in_channels=3,
                                   width=32,
                                   blocks=[1,2]).to(device)

model = VQVAE(in_dim=3,
              out_dim=16, 
              h_dim=32, 
              num_res_layers=[2, 2, 2, 2, 2], 
              num_embeddings=512, 
              embedding_dim=128,
              decay=None,
              commitment_cost=0.25).to(device)

load_model_weights(model, 'weights/vqvae_exp10.pt')
load_model_weights(channel_compressor, 'weights/mixer_exp10.pt')
model.train()
channel_compressor.train()
feature_extractor.train()
optimizer = torch.optim.Adam([{'params':channel_compressor.parameters(), 'lr':0.001}, 
                              {'params':model.parameters(), 'lr':0.001}, 
                              {'params':feature_extractor.parameters(), 'lr':0.001}])
num_epochs = 1000
len_loader = len(train_loader)
# channel_compressor.freeze_model()

for epoch in range(1, num_epochs + 1):
    train_mse_loss = 0
    train_vq_loss = 0
    train_ssim_loss = 0
    train_perplexity = 0
    # if epoch == 20:
    #     channel_compressor.unfreeze_model()
    #     print('--- channel_compressor unfreezed! ---')
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=False)
    for x in progress_bar:
        x = x.to(device)
        x_noised = reduce_transform(noise_transform(x))
        optimizer.zero_grad()
        # mixing
        x_compr = channel_compressor(x_noised)
        # random huv transformation
        x_compr_hue = hue_transform(x_compr)
        # forward
        x_recon, vq_loss, perplexity, quantized = model(x_compr) # teacher model
        feature = feature_extractor(x_compr_hue) # student model
        print(quantized.size(), feature.size())
        # compute loss
        mse_loss = F.mse_loss(x_recon, x)
        ssim_loss = 1 - ssim(x_recon, x, data_range=1, size_average=True)
        pifm_loss = F.mse_loss(feature, quantized)
        loss = mse_loss + 0.05*ssim_loss + 0.02*vq_loss + 0.1*pifm_loss
        loss.backward()
        optimizer.step()
        # image saving
        
        train_mse_loss += mse_loss.item()
        train_vq_loss += vq_loss.item()
        train_ssim_loss += ssim_loss.item()
        train_perplexity += perplexity.item()
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        coeff = channel_compressor.coeff.cpu().detach().numpy()
        progress_bar.set_postfix({
            'GPU': f"{memory_allocated:.0f} MB",
            'MSE loss': f"{mse_loss.item():.5f}",
            'VQ loss':  f"{vq_loss.item():.5f}",
            'SSIM loss':f"{ssim_loss.item():.5f}",
            'Perplexity': f"{perplexity.item():.5f}",
            'Coeff': f"{coeff}",
            })
    
    test_x_compr = channel_compressor(test_x)
    test_x_recon, _, _, _ = model(test_x_compr)
    save_mix_image(test_x_compr, epoch, exp_path, intervals=20)
    save_images(test_x_recon, epoch, exp_path, intervals=20)
    save_model(model, 'vqvae', epoch, exp_path, intervals=20)
    save_model(feature_extractor, 'pifm', epoch, exp_path, intervals=20)
    save_model(channel_compressor, 'mixer', epoch, exp_path, intervals=50)
    if epoch>=0:
        save_weights(epoch+1, channel_compressor.conv1x1, os.path.join(exp_path,'weights.txt'))
    print('Epoch: {} | MSE loss: {:.5f} | VQ loss: {:.4f} | SSIM loss: {:.4f} | Perplexity: {:.4f}'.format(epoch, 
                                                                                                            train_mse_loss / len_loader,
                                                                                                            train_vq_loss / len_loader,
                                                                                                            train_ssim_loss / len_loader,
                                                                                                            train_perplexity / len_loader,
                                                                                                            ))