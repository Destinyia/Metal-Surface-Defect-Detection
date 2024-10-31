import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from PIL import Image, ImageDraw
# from loss import compute_targets

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetBackbone, self).__init__()
        # Using the layers until layer4
        self.stem = nn.Sequential(*list(resnet_model.children())[:4])  # Includes conv1, bn1, relu, maxpool
        self.layer1 = resnet_model.layer1  # ResNet C2
        self.layer2 = resnet_model.layer2  # ResNet C3
        self.layer3 = resnet_model.layer3  # ResNet C4
        self.layer4 = resnet_model.layer4  # ResNet C5

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.top_block = nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        p3 = self.output_convs[0](p3)
        p4 = self.output_convs[1](p4)
        p5 = self.output_convs[2](p5)
        # p6 = self.top_block(c5)
        
        return [p3, p4, p5]#, p6]

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCOSHead, self).__init__()
        self.cls_tower = self._make_tower(in_channels)
        self.bbox_tower = self._make_tower(in_channels)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def _make_tower(self, in_channels, num_convs=4):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        centernesses = []

        for feature in x:
            cls_tower_out = self.cls_tower(feature)
            bbox_tower_out = self.bbox_tower(feature)
            cls_logits.append(self.cls_logits(cls_tower_out))
            bbox_preds.append(self.bbox_pred(bbox_tower_out))
            centernesses.append(self.centerness(bbox_tower_out))

        return cls_logits, bbox_preds, centernesses

class FCOS(nn.Module):
    def __init__(self, num_classes, backbone=None):
        super(FCOS, self).__init__()
        if backbone is None:
            backbone = models.resnet34(weights="DEFAULT")
        self.backbone = ResNetBackbone(backbone)
        self.fpn = FPN([128, 256, 512], 256)
        self.head = FCOSHead(256, num_classes)

    def forward(self, x):
        backbone_feats = self.backbone(x)
        fpn_feats = self.fpn(backbone_feats)
        cls_logits, bbox_preds, centernesses = self.head(fpn_feats)
        return cls_logits, bbox_preds, centernesses



if __name__ == "__main__":
    # Initialize the ResNet34 model
    resnet_model = models.resnet34(pretrained=True)
    
    # Initialize the Backbone
    backbone = ResNetBackbone(resnet_model)
    
    # Example input tensor (batch_size=2, channels=3, height=512, width=512)
    input_tensor = torch.randn(2, 3, 512, 512)
    
    # Forward pass through the backbone
    c3, c4, c5 = backbone(input_tensor)
    
    # Print the shapes of the backbone outputs
    print("Backbone outputs:")
    print(f"C3 shape: {c3.shape}")
    print(f"C4 shape: {c4.shape}")
    print(f"C5 shape: {c5.shape}")
    
    # Initialize the FPN
    fpn = FPN([128, 256, 512], 256)
    
    # Forward pass through the FPN
    fpn_outputs = fpn((c3, c4, c5))
    
    # Print the shapes of the FPN outputs
    print("\nFPN outputs:")
    for i, fpn_output in enumerate(fpn_outputs):
        print(f"P{i+3} shape: {fpn_output.shape}")
    
    num_classes = 7

    # Initialize the FCOS model
    fcos_model = FCOS(num_classes)

    # Example input tensor (batch_size=2, channels=3, height=512, width=512)
    input_tensor = torch.randn(2, 3, 512, 512)

    # Example targets (list of ground truth boxes for each image in the batch)
    targets = [...]  # Define your targets here

    # Forward pass
    logits, bbox_pred, centernesses = fcos_model(input_tensor)
    for i in range(len(logits)):
        print(logits[i].shape, bbox_pred[i].shape, centernesses[i].shape)