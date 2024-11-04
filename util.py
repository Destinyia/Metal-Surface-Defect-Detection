import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_images(tensor):
    """
    将模型的输出Tensor转换为图像列表。
    参数:
    - tensor (torch.Tensor): 应该是一个四维的Tensor，例如[batch_size, channels, height, width]
    
    返回:
    - images (list): 包含PIL图像的列表
    """
    images = tensor.detach().cpu()  # 确保tensor在CPU上，并且无需梯度计算
    images = images.permute(0, 2, 3, 1)  # 重排维度为[batch_size, height, width, channels]
    images = [img.numpy() for img in images]  # 转换为numpy数组
    return images

def tensor_to_16images(tensor):
    """
    将模型的输出Tensor转换为图像列表。
    参数:
    - tensor (torch.Tensor): 应该是一个四维的Tensor，例如[batch_size, channels, height, width]
    
    返回:
    - images (list): 包含PIL图像的列表
    """
    images = tensor.detach().cpu()  # 确保tensor在CPU上，并且无需梯度计算
    images = images.permute(1, 2, 0)  # 重排维度为[batch_size, height, width, channels]
    images = [images[:,:,i:i+1].numpy() for i in range(16)]  # 转换为numpy数组
    return images

def show_images(images, cols=4):
    """
    使用matplotlib显示图像列表。
    参数:
    - images (list): 图像列表
    - cols (int): 每行显示的图像数量
    """
    rows = (len(images) + cols - 1) // cols  # 计算需要多少行
    fig = plt.figure(figsize=(cols * 10, rows * 10))  # 创建一个足够大的图形
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image, interpolation='nearest')  # 显示图像
        ax.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    plt.show()
    
def show_16_images(images, cols=4):
    """
    使用matplotlib显示图像列表。
    参数:
    - images (list): 图像列表
    - cols (int): 每行显示的图像数量
    """
    rows = (len(images) + cols - 1) // cols  # 计算需要多少行
    fig = plt.figure(figsize=(cols * 4, rows * 4))  # 创建一个足够大的图形
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image, interpolation='nearest')  # 显示图像
        ax.axis('off')  # 不显示坐标轴
    plt.show()

def save_mix_image(tensor, epoch, exp_path='runs', intervals=10):
    tensor = tensor[0].detach()
    tensor -= tensor.min()
    tensor /= tensor.max()
    image = tensor.cpu().permute(1, 2, 0).numpy()
    image = (image*255.0).astype(np.uint8)
    save_epoch = epoch - epoch % intervals
    save_epoch = f'epoch{save_epoch}-{save_epoch+intervals-1}.jpg'
    cv2.imwrite(os.path.join(exp_path, save_epoch), image)

def save_images(tensor, epoch, exp_path='runs', intervals=10):
    """
    保存最后一次预测的tensor为4x4拼接图像。

    参数:
    tensor - 预测输出的Tensor，形状应为(batch_size, channels, height, width)
    epoch - 当前epoch编号，用于文件命名
    path - 图像保存路径
    """

    # 选择tensor中的前16个通道, 缩放到[0,1]
    tensor = tensor[0].detach()
    tensor -= tensor.min()
    tensor /= tensor.max()
    images = tensor.cpu().numpy()  # 取第一个样本的前16个通道

    # 创建一个4x4的子图
    fig, axes = plt.subplots(4, 4, figsize=(32, 32))
    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 不显示坐标轴

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)  # 调整子图布局
    # 保存图像
    save_epoch = epoch - epoch % intervals
    save_epoch = f'epoch{save_epoch}-{save_epoch+intervals-1}.png'
    plt.savefig(os.path.join(exp_path, save_epoch))
    plt.close()

def save_weights(epoch, layer, path):
    with open(path, 'a') as f:
        weights = layer.weight.data.reshape(16,).cpu().numpy().round(4)
        bias = layer.bias.data.cpu().numpy().round(4)
        # f.write(f'Epoch {epoch} bias:{bias[0]},{bias[1]},{bias[2]} - weights:\n')
        # for i in range(3):
        #     f.write(f'{epoch},{i},'+','.join([str(x) for x in weights[i]])+'\n')
        f.write(f'{epoch},'+','.join([str(x) for x in weights])+'\n')

def save_model(model, name, epoch, exp_path='runs/exp', intervals=20):
    save_epoch = epoch - epoch % intervals
    save_epoch = f'{name}_epoch{save_epoch}-{save_epoch+intervals-1}.pt'
    torch.save(model.state_dict(), os.path.join(exp_path, save_epoch))

def load_model_weights(model, weight_path):
    try:
        # 加载权重文件
        state_dict = torch.load(weight_path)
        result = model.load_state_dict(state_dict, strict=False)
        
        # 检查未加载和多余的键
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
        
        if missing_keys:
            print("以下层未能加载权重：", missing_keys)
        if unexpected_keys:
            print("权重文件中包含未使用的键：", unexpected_keys)
        
    except Exception as e:
        print("加载权重时出错：", str(e))
