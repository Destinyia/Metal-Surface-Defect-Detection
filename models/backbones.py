import torch
import torch.nn as nn

# Focus Layer (used to slice input into 4 parts for better spatial info)
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # Slicing the input into 4 parts
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))

# Define the C3 Module (used in YOLOv5)
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(C3, self).__init__()
        self.split_channels = in_channels // 2
        # One part of the input goes through several residual blocks
        self.blocks = nn.Sequential(
            *[nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, stride=1, bias=False) for _ in range(num_blocks)],
            nn.BatchNorm2d(self.split_channels),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1, x2 = torch.split(x, self.split_channels, dim=1)  # Split input channels
        x1 = self.blocks(x1)  # Process half of the input
        x = torch.cat([x1, x2], dim=1)  # Concatenate the processed and skipped parts
        return self.relu(self.bn(self.conv1x1(x)))  # Apply final conv, batch norm, and activation

# Define the C2f Module
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super(C2f, self).__init__()
        self.split_channels = in_channels // 2
        self.blocks = nn.Sequential(
            *[nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, stride=1) for _ in range(num_blocks)]
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1, x2 = torch.split(x, self.split_channels, dim=1)  # Split the input channels
        x1 = self.blocks(x1)  # Pass only half of the input through the blocks
        x = torch.cat([x1, x2], dim=1)  # Concatenate the split parts
        return self.conv1x1(x)  # Final 1x1 conv to unify channels

# Define the YOLOv8 Backbone
class YOLOv5Backbone(nn.Module):
    def __init__(self, in_channels=3, width=32, blocks=[1, 2]):
        # YOLOv8s depth: 0.33 width: 0.5
        super(YOLOv5Backbone, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),  # Initial downsampling
            nn.BatchNorm2d(width),
            nn.ReLU()
        )
        
        # Stages with C2f Modules
        self.p2 = nn.Sequential(
            nn.Conv2d(width, width*2, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(width*2),
            nn.ReLU(),
            C3(width*2, width*2, num_blocks=blocks[0])
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(width*2, width*4, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(width*4),
            nn.ReLU(),
            C3(width*4, width*4, num_blocks=blocks[1])
        )

    def forward(self, x):
        x = self.p1(x)
        x1 = self.p2(x)
        x2 = self.p3(x1)
        return x2  # Return the final feature map for the next layer in the network

# Define the YOLOv8 Backbone
class YOLOv8Backbone(nn.Module):
    def __init__(self, in_channels=3, width=32, blocks=[1, 2]):
        # YOLOv8s depth: 0.33 width: 0.5
        super(YOLOv8Backbone, self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),  # Initial downsampling
            nn.BatchNorm2d(width),
            nn.ReLU()
        )
        
        # Stages with C2f Modules
        self.p2 = nn.Sequential(
            nn.Conv2d(width, width*2, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(width*2),
            nn.ReLU(),
            C2f(width*2, width*2, num_blocks=blocks[0])
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(width*2, width*4, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.BatchNorm2d(width*4),
            nn.ReLU(),
            C2f(width*4, width*4, num_blocks=blocks[1])
        )

    def forward(self, x):
        x = self.p1(x)
        x1 = self.p2(x)
        x2 = self.p3(x1)
        return x2  # Return the final feature map for the next layer in the network

# Define the HGStem Module (used in RT-DETR)
class HGStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HGStem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Define the HGBlock Module (Hourglass Block)
class HGBlock(nn.Module):
    def __init__(self, channels):
        super(HGBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

# Define the Depthwise Convolution Module (DWConv)
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return self.relu(self.bn(x))


# Define the RT-DETR Backbone
class RTDETRBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(RTDETRBackbone, self).__init__()
        # Initial Stem Layer
        self.p1 = HGStem(in_channels, 64)
        # Hourglass Blocks
        self.p2 = HGBlock(64)
        self.p3 = nn.Sequential(
            DWConv(64, 128, stride=2),
            HGBlock(128)
        )
        # self.p4 = nn.Sequential(
        #     DWConv(128, 256, stride=2),
        #     HGBlock(256)
        # )
        # self.p5 = nn.Sequential(
        #     DWConv(256, 512, stride=2),
        #     HGBlock(512)
        # )

    def forward(self, x):
        x = self.p1(x)  # Initial stem layer
        x1 = self.p2(x)  # First hourglass block
        x2 = self.p3(x1)  # Second stage with DWConv and HGBlock
        # x3 = self.p4(x2)  # Third stage with DWConv and HGBlock
        # x4 = self.p5(x3)  # Fourth stage with DWConv and HGBlock
        return x2  # Return the final feature map for further layers in the network

# Test the Backbone
if __name__ == "__main__":
    model = YOLOv8Backbone()
    x = torch.randn(1, 3, 256, 256)  # Dummy input
    output = model(x)
    print("Output shape:", output.shape)  # Should show (1, 512, 16, 16) if input is (1, 3, 256, 256)