import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CLASS_NAMES = {
    0: 'BIODEGRADABLE',
    1: 'CARDBOARD',
    2: 'GLASS',
    3: 'METAL',
    4: 'PAPER',
    5: 'PLASTIC'
}

COLORS = [
    (0, 255, 0), (150, 255, 0), (0, 0, 255),
    (255, 0, 0), (255, 0, 255), (0, 255, 255)
]


architecture_config = [ 
    (7, 64, 2, 3),      
    "M",                
    (3, 128, 1, 1),     
    "M",               
    (3, 256, 1, 1),     
    "M",                
    (3, 512, 1, 1),     
    "M",                
    (3, 512, 1, 1),     
    (3, 512, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs_(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, 
                        out_channels=x[1], 
                        kernel_size=x[0], 
                        stride=x[2], 
                        padding=x[3],
                    )
                ]
                in_channels = x[1]
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
              
        return nn.Sequential(*layers)
        
    def _create_fcs_(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(512*S*S, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, S*S*(C+B*5)),
        )



def cellboxes_to_boxes(predictions, S=14):
    """
    Convert YOLO grid predictions to bounding boxes
    predictions: (batch_size, S*S*(C+B*5))
    Returns: tensor of boxes [class, confidence, x, y, w, h]
    """
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, -1)
    
    # Box predictions 
    bboxes1 = predictions[..., 7:11]   # First box: [x, y, w, h]
    bboxes2 = predictions[..., 12:16]  # Second box: [x, y, w, h]
    
    # Confidence scores
    scores1 = predictions[..., 6:7]    # First box confidence
    scores2 = predictions[..., 11:12]  # Second box confidence
    
    # Choosing box with higher confidence
    scores = torch.cat((scores1, scores2), dim=-1)
    best_box = scores.argmax(-1).unsqueeze(-1).float()  # [batch, S, S, 1]
    
    # best box coordinates
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    
  
    cell_indices_x = torch.arange(S, device=predictions.device).repeat(batch_size, S, 1).unsqueeze(-1)
    cell_indices_y = torch.arange(S, device=predictions.device).repeat(batch_size, S, 1).unsqueeze(-1).permute(0, 2, 1, 3)
    
    
    x = (best_boxes[..., 0:1] + cell_indices_x) / S
    y = (best_boxes[..., 1:2] + cell_indices_y) / S
    
    w = best_boxes[..., 2:3] / S
    h = best_boxes[..., 3:4] / S
    
    converted_bboxes = torch.cat((x, y, w, h), dim=-1)
    
    # Get predicted class and confidence
    predicted_class = predictions[..., :6].argmax(-1).unsqueeze(-1).float()
    best_confidence = torch.max(scores1, scores2)
    
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in midpoint format [x, y, w, h]
    """
    # Convert to corners
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection + 1e-6
    
    return intersection / union

def non_max_suppression(bboxes, iou_threshold=0.5, confidence_threshold=0.4):
    """
    Apply NMS to remove overlapping boxes
    bboxes: list of [class, confidence, x, y, w, h]
    """
    # Filter by confidence
    bboxes = [box for box in bboxes if box[1] > confidence_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        # Keep boxes with different class OR low IoU overlap
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]  # Different class
            or calculate_iou(chosen_box[2:], box[2:]) < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

def draw_boxes(image, boxes, class_names, colors):
    """
    Draw bounding boxes on image
    """
    h, w = image.shape[:2]
    
    for box in boxes:
        class_id = int(box[0])
        confidence = box[1]
        x_center, y_center, width, height = box[2:]
        
        # Convert from normalized to pixel coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw rectangle
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_names[class_id]}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image, (x1, y1 - text_height - 8), 
            (x1 + text_width, y1), color, -1
        )
        cv2.putText(
            image, label, (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return image

def predict_image(model, image_path, confidence_threshold=0.3, iou_threshold=0.5, S=14):
    """
    Run inference on a single image
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Resize to 448x448 (same as training)
    image_resized = cv2.resize(image, (448, 448))
    image_tensor = T.ToTensor()(image_resized).unsqueeze(0).to(DEVICE)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Convert to boxes with S=14
    boxes = cellboxes_to_boxes(predictions, S=S)
    boxes = boxes[0].cpu().numpy()  # First image in batch
    
    # Convert to list format for NMS
    boxes_list = []
    for i in range(S):
        for j in range(S):
            box = boxes[i, j]
            if box[1] > confidence_threshold:  # Check confidence
                boxes_list.append(box.tolist())
    
    # Apply NMS
    boxes_nms = non_max_suppression(boxes_list, iou_threshold, confidence_threshold)
    
    # Draw boxes
    result_image = draw_boxes(original_image, boxes_nms, CLASS_NAMES, COLORS)
    
    return result_image, boxes_nms



def main():
    print(f"[INFO] Using device: {DEVICE}")
    
    
    model = Yolov1(split_size=14, num_boxes=2, num_classes=6).to(DEVICE)
    
    #
    checkpoint_path = "/Users/lavanyanigam/Desktop/SortNow/best_model.pth"  
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"[INFO] Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"[INFO] Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        else:
            
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded model state dict")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    model.eval()
    
    
    image_path = "/Users/lavanyanigam/Desktop/SortNow/testimages/Screenshot 2025-12-26 at 9.25.55 AM.png"  # image path 
    
    try:
        result_image, boxes = predict_image(
            model, image_path, 
            confidence_threshold=0.30, 
            iou_threshold=0.5,
            S=14  
        )
        
        # Display results
        print(f"\n{'='*50}")
        print(f"Detected {len(boxes)} objects:")
        print(f"{'='*50}")
        for i, box in enumerate(boxes, 1):
            class_id = int(box[0])
            confidence = box[1]
            x, y, w, h = box[2:]
            print(f"{i}. {CLASS_NAMES[class_id]}: {confidence:.2%} | Box: ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")
        print(f"{'='*50}\n")
        
        # Save result
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        output_path = "/Users/lavanyanigam/Desktop/SortNow/results/predictionresult0.jpg"
        cv2.imwrite(output_path, result_image_bgr)
        print(f"[INFO] Result saved to {output_path}")
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
