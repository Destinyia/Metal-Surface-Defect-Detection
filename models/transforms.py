import torch
import numpy as np
import random

class EllipseNoise:
    def __init__(self, num_ellipses=5, dark=0.9, noise_level=0.5):
        """
        :param center: 椭圆的中心 (cx, cy)
        :param axes: 椭圆的长轴和短轴 (a, b)
        :param noise_level: 噪声强度，0-1之间
        """
        self.dark = dark
        self.num_ellipses = num_ellipses
        self.noise_level = noise_level

    def __call__(self, img):
        """
        :param img: 输入图像张量 (C, H, W)
        :return: 添加噪声后的图像张量
        """
        # 获取图像形状
        B, C, H, W = img.shape

        # 创建全为1的mask
        mask = torch.ones((H, W), dtype=torch.float32, device=img.device)

        # 生成椭圆形区域内的噪声点
        
        for _ in range(self.num_ellipses):
            # 随机生成椭圆的中心、长轴和短轴
            cx = random.randint(200, W - 200)
            cy = random.randint(200, H - 200)
            a = random.randint(40, 160)  # 长轴
            b = random.randint(40, 160)  # 短轴

            # 生成椭圆形区域的掩码
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            ellipse_mask = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1
            noise = torch.rand(H, W, device=img.device) * self.noise_level + (self.dark - self.noise_level)
            mask[ellipse_mask] = noise[ellipse_mask]

        # 将 mask 应用于输入图像张量
        img = img * mask

        return img
    
class ChannelReduce:
    def __init__(self, reduce_range=(0.4, 0.8)):
        """
        :param reduce_range: 亮度降低的范围
        """
        self.reduce_range = reduce_range

    def __call__(self, img):
        """
        :param img: 输入图像张量 (B, C, H, W)
        :return: 修改后的图像张量
        """
        # 获取图像形状
        B, C, H, W = img.shape

        # 对每个样本处理
        for b in range(B):
            # 随机选择一个通道
            chosen_channel = random.randint(0, C - 1)
            # 随机选择一个降低亮度的系数
            reduce_factor = random.uniform(*self.reduce_range)
            # 将选择的通道亮度降低
            img[b, chosen_channel, :, :] *= reduce_factor

        return img

# 示例用法
if __name__ == "__main__":
    # 创建一个示例图像张量 (C, H, W)
    img = torch.ones((1, 3, 1600, 1600), dtype=torch.float32, device='cuda')

    # 创建变换实例
    transform = EllipseNoiseTransform(num_ellipses=6, dark=0.96, noise_level=0.5)

    # 应用变换
    transformed_img = transform(img)

    # 可视化结果
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img[0][0].cpu().numpy(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(transformed_img[0][0].detach().cpu().numpy(), cmap='gray')
    ax[1].set_title('Transformed Image with Ellipse Noise')
    plt.show()