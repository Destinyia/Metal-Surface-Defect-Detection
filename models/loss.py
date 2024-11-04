import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torchvision.ops import sigmoid_focal_loss

def compute_iou(boxes1, boxes2):
    """
    Calculate the IoU between each box in boxes1 and boxes2.
    boxes1: (N, 4), boxes2: (M, 4)
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def compute_targets(images, logits, bbox_pred, targets, strides, k=7):
    """
    Compute targets for FCOS using ATSS sampling strategy.
    """
    cls_targets = []
    bbox_targets = []
    centerness_targets = []
    
    # 每一个尺度
    for logit, bbox, stride in zip(logits, bbox_pred, strides):
        # print(targets, bbox.shape, stride)
        N, C, H, W = logit.shape
        A = bbox.shape[1] // 4  # number of anchors per location
        device = logit.device
        
        # Generate grid of coordinates
        yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        xv = xv.to(device) * stride
        yv = yv.to(device) * stride
        grid = torch.stack((xv, yv, xv+stride, yv+stride), -1).float()  # (H, W, 2)
        grid = grid.reshape(-1, 4)  # (H*W, 4)， 每一层的anchor, x1y1x2y2
        grid_centers = grid[:, :2] + stride // 2
        grid_centers = grid_centers.reshape(-1, 2) # (H*W, 2)

        # Initialize targets
        cls_target = torch.zeros((N, C, H * W), dtype=torch.long, device=device)
        bbox_target = torch.zeros((N, H * W, 4), dtype=torch.float32, device=device)
        centerness_target = torch.zeros((N, H * W), dtype=torch.float32, device=device)

        for i in range(N):
            # TODO: only consider boxes within the scale range
            
            # labels & boxes
            labels = targets[i][:, 0].int()
            boxes = targets[i][:, 1:]
            num_gt = labels.shape[0]
            if num_gt == 0:
                continue
            
            # Compute IoU between anchors and GT boxes
            iou_matrix = compute_iou(grid, boxes.to(device))
            
            # ATSS: compute L2 distance between gt & grid center
            gt_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            distances = torch.cdist(grid_centers, gt_centers)
            
            # ATSS: select top-k IoUs per GT box
            topk = min(k, iou_matrix.size(0))
            topk_dis, topk_indices = torch.topk(distances, topk, dim=0, largest=False)
            topk_indices_y = torch.arange(boxes.shape[0]).repeat(topk, 1)
            iou_topk = iou_matrix[topk_indices, topk_indices_y]
            
            # iou_topk, _ = torch.topk(iou_matrix, topk, dim=0)
            iou_mean = iou_topk.mean(dim=0)
            iou_std = iou_topk.std(dim=0)
            iou_thresh = torch.clip(iou_mean + iou_std*0.5, 0.05)
            
            # Select positive samples
            is_pos = iou_matrix >= iou_thresh.unsqueeze(0)
            
            # Only consider anchors inside the GT box
            l = grid[:, 0].unsqueeze(1) - boxes[:, 0] + stride//2
            t = grid[:, 1].unsqueeze(1) - boxes[:, 1] + stride//2
            r = boxes[:, 2] - grid[:, 0].unsqueeze(1) - stride//2
            b = boxes[:, 3] - grid[:, 1].unsqueeze(1) - stride//2
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            is_pos = is_pos & is_in_boxes
            
            # Select the highest IoU per anchor
            iou_matrix[~is_pos] = -1
            iou_max, iou_max_indices = iou_matrix.max(dim=1)
            positive_indices = torch.where(iou_max > 0)[0]
            
            # 构建target
            cls_target[i, labels[iou_max_indices[positive_indices]], positive_indices] = 1
            bbox_target[i, positive_indices] = reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], :]
            centerness_target[i, positive_indices] = torch.sqrt(
                (torch.min(reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 0], reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 2]) / 
                 torch.max(reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 0], reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 2])) * 
                (torch.min(reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 1], reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 3]) / 
                 torch.max(reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 1], reg_targets_per_im[positive_indices, iou_max_indices[positive_indices], 3])))
            # print(centerness_target[i].view(H, W))
            
            # 绘制网格
            # draw_grid(images, cls_target, i, stride, boxes, C, H, W)

        cls_targets.append(cls_target.view(N, C, H, W))
        bbox_targets.append(bbox_target.view(N, H, W, 4))
        centerness_targets.append(centerness_target.view(N, H, W))
    
    return cls_targets, bbox_targets, centerness_targets

def draw_grid(images, cls_target, i, stride, boxes, C, H, W):
    cls_grid = cls_target[i].view(C, H, W)
    # print(cls_grid[labels[i].int()])
    out_image = Image.fromarray((images[i].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
    draw = ImageDraw.Draw(out_image)
    for x in range(0, W):
        draw.line([(x*stride, 0), (x*stride, 640)], fill="gray", width=1)
    for y in range(0, H):
        draw.line([(0, y*stride), (640, y*stride)], fill="gray", width=1)    
    for x in range(0, W):
        for y in range(0, H):
            for j in range(C):
                if cls_grid[j, y, x] == 1:
                    draw.rectangle([x*stride, y*stride, (x+1)*stride, (y+1)*stride], outline="yellow", width=2)
    for bbox in boxes:
        draw.rectangle(torch.round(bbox).int().tolist(), outline="red", width=2)
    out_image.save(f'b{i}_s{stride}.jpg')
    

def ciou_loss(preds, targets):
    """
    Complete IoU (CIoU) loss implementation.
    Args:
        preds: Predicted bounding boxes, shape (N, 4).
        targets: Target bounding boxes, shape (N, 4).
    Returns:
        CIoU loss, shape (N,).
    """
    pred_left = preds[:, 0]
    pred_top = preds[:, 1]
    pred_right = preds[:, 2]
    pred_bottom = preds[:, 3]
    
    target_left = targets[:, 0]
    target_top = targets[:, 1]
    target_right = targets[:, 2]
    target_bottom = targets[:, 3]
    
    pred_widths = pred_right - pred_left
    pred_heights = pred_bottom - pred_top
    pred_ctr_x = (pred_left + pred_right) / 2
    pred_ctr_y = (pred_top + pred_bottom) / 2

    target_widths = target_right - target_left
    target_heights = target_bottom - target_top
    target_ctr_x = (target_left + target_right) / 2
    target_ctr_y = (target_top + target_bottom) / 2

    inter_left = torch.max(pred_left, target_left)
    inter_top = torch.max(pred_top, target_top)
    inter_right = torch.min(pred_right, target_right)
    inter_bottom = torch.min(pred_bottom, target_bottom)
    
    inter_widths = (inter_right - inter_left).clamp(min=0)
    inter_heights = (inter_bottom - inter_top).clamp(min=0)
    inter_area = inter_widths * inter_heights
    
    pred_area = pred_widths * pred_heights
    target_area = target_widths * target_heights
    
    union_area = pred_area + target_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    
    enclose_left = torch.min(pred_left, target_left)
    enclose_top = torch.min(pred_top, target_top)
    enclose_right = torch.max(pred_right, target_right)
    enclose_bottom = torch.max(pred_bottom, target_bottom)
    
    enclose_width = enclose_right - enclose_left
    enclose_height = enclose_bottom - enclose_top
    enclose_diagonal = enclose_width ** 2 + enclose_height ** 2
    
    center_distance = (pred_ctr_x - target_ctr_x) ** 2 + (pred_ctr_y - target_ctr_y) ** 2
    
    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(target_widths / target_heights) - torch.atan(pred_widths / pred_heights)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v).clamp(min=1e-6)
    
    ciou = iou - (center_distance / enclose_diagonal.clamp(min=1e-6)) - alpha * v
    return 1 - ciou

def compute_loss(logits_targets, bbox_targets, centerness_targets, logits, bbox_pred, centernesses):
    """
    Compute FCOS losses.
    Args:
        logits_targets: List of classification targets from compute_targets, each element of shape (N, C, H, W).
        bbox_targets: List of bbox regression targets from compute_targets, each element of shape (N, H, W, 4).
        centerness_targets: List of centerness targets from compute_targets, each element of shape (N, H, W).
        logits: List of classification logits from FCOS, each element of shape (N, C, H, W).
        bbox_pred: List of bbox regression predictions from FCOS, each element of shape (N, 4, H, W).
        centernesses: List of centerness predictions from FCOS, each element of shape (N, 1, H, W).
    Returns:
        Total loss, classification loss, bbox regression loss, centerness loss.
    """
    # Initialize total losses and total number of positive samples
    total_cls_loss = torch.tensor(0.0, device=logits[0].device)
    total_bbox_loss = torch.tensor(0.0, device=logits[0].device)
    total_centerness_loss = torch.tensor(0.0, device=logits[0].device)
    total_num_pos = torch.tensor(0.0, device=logits[0].device)

    for logits_target, bbox_target, centerness_target, logit, bbox, centerness in zip(
            logits_targets, bbox_targets, centerness_targets, logits, bbox_pred, centernesses):

        # Reshape logits and targets to 1D tensors
        N, C, H, W = logit.shape
        logit = logit.permute(0, 2, 3, 1).reshape(-1, C)
        logits_target = logits_target.permute(0, 2, 3, 1).reshape(-1, C)

        # Create a mask for positive samples
        positive_mask = logits_target > 0
        num_pos = positive_mask.sum().float()

        # Compute classification loss
        total_cls_loss += sigmoid_focal_loss(logit, logits_target.float(), reduction='mean')
        
        bbox = bbox.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_target = bbox_target.view(-1, 4)

        if num_pos > 0:
            total_num_pos += num_pos
            pos_idx = positive_mask.nonzero()[:,0]
            total_bbox_loss += ciou_loss(bbox[pos_idx], bbox_target[pos_idx]).sum() #/ num_pos.clamp(min=1.0))
            total_centerness_loss += F.binary_cross_entropy_with_logits(
                centerness.view(-1)[pos_idx], centerness_target.view(-1)[pos_idx].float(), reduction='sum')
        # else:
        #     centerness_loss.append(torch.tensor(0.0, device=bbox.device))
            
    # total_cls_loss = total_cls_loss / len(logits)
    total_bbox_loss = total_bbox_loss / total_num_pos.clamp(min=1.0)
    total_centerness_loss = total_centerness_loss / total_num_pos.clamp(min=1.0)

    total_loss = total_cls_loss + total_bbox_loss + total_centerness_loss

    return total_loss, total_cls_loss, total_bbox_loss, total_centerness_loss