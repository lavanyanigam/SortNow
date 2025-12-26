import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    union = box1_area + box2_area - intersection + 1e-6

    return intersection/union


def save_checkpoint(state, filename="checkpoint.pth"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def non_max_suppression(predictions, conf_threshold=0.3, iou_threshold=0.45, S=7, B=2, C=6):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes
    
    Args:
        predictions: Model output tensor [S*S*(C+B*5)]
        conf_threshold: Minimum confidence to keep detection
        iou_threshold: IoU threshold for NMS
        S: Grid size (7x7)
        B: Number of boxes per cell (2)
        C: Number of classes (6)
    
    Returns:
        List of filtered detections with class, confidence, bbox
    """
    predictions = predictions.reshape(S, S, C + B*5)
    boxes = []
    
    for i in range(S):
        for j in range(S):
            for b in range(B):
                if b == 0:
                    confidence = predictions[i, j, 6].item()
                    bbox = predictions[i, j, 7:11]
                else:
                    confidence = predictions[i, j, 11].item()
                    bbox = predictions[i, j, 12:16]
                
                if confidence > conf_threshold:
                    class_probs = predictions[i, j, :C]
                    class_idx = torch.argmax(class_probs).item()
                    class_conf = class_probs[class_idx].item()
                    
                    x_cell, y_cell, w_cell, h_cell = bbox
                    x_center = (j + x_cell.item()) / S
                    y_center = (i + y_cell.item()) / S
                    width = w_cell.item() / S
                    height = h_cell.item() / S
                    
                    x1 = max(0, x_center - width / 2)
                    y1 = max(0, y_center - height / 2)
                    x2 = min(1, x_center + width / 2)
                    y2 = min(1, y_center + height / 2)
                    
                    boxes.append({
                        'class': class_idx,
                        'confidence': confidence * class_conf,
                        'bbox': [x1, y1, x2, y2]
                    })
    
    boxes.sort(key=lambda x: x['confidence'], reverse=True)
    
    final_boxes = []
    while boxes:
        best_box = boxes.pop(0)
        final_boxes.append(best_box)
        
        boxes = [box for box in boxes if 
                 compute_iou(best_box['bbox'], box['bbox']) < iou_threshold or 
                 box['class'] != best_box['class']]
    
    return final_boxes

def compute_iou(box1, box2):
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)
