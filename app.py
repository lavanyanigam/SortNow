
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

CLASS_NAMES = {0: 'BIODEGRADABLE', 1: 'CARDBOARD', 2: 'GLASS', 3: 'METAL', 4: 'PAPER', 5: 'PLASTIC'}
BINS = {0: 'Green Bin', 1: 'Blue Bin', 2: 'Blue Bin', 3: 'Blue Bin', 4: 'Blue Bin', 5: 'Blue Bin'}


architecture_config = [(7,64,2,3), "M", (3,128,1,1), "M", (3,256,1,1), "M", (3,512,1,1), "M", (3,512,1,1), (3,512,1,1)]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=14, num_boxes=2, num_classes=6):
        super(Yolov1, self).__init__()
        self.darknet = self._create_conv_layers(architecture_config, in_channels)
        self.fcs = nn.Sequential(
            nn.Linear(512*split_size*split_size, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(2048, split_size*split_size*(num_classes+num_boxes*5))
        )

    def forward(self, x):
        return self.fcs(torch.flatten(self.darknet(x), start_dim=1))

    def _create_conv_layers(self, config, in_channels):
        layers = []
        for x in config:
            if type(x) == tuple:
                layers.append(CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

def cellboxes_to_boxes(pred, S=14):
    pred = pred.reshape(-1, S, S, 16)
    best_box = torch.cat([pred[...,6:7], pred[...,11:12]], dim=-1).argmax(-1).unsqueeze(-1).float()
    best_boxes = pred[...,7:11] * (1 - best_box) + pred[...,12:16] * best_box
    
    cell_x = torch.arange(S).repeat(1, S, 1).unsqueeze(-1)
    cell_y = cell_x.permute(0, 2, 1, 3)
    
    boxes = torch.cat([
        pred[..., :6].argmax(-1).unsqueeze(-1).float(),
        torch.max(pred[...,6:7], pred[...,11:12]),
        (best_boxes[...,0:1] + cell_x) / S,
        (best_boxes[...,1:2] + cell_y) / S,
        best_boxes[...,2:3] / S,
        best_boxes[...,3:4] / S
    ], dim=-1)
    
    return boxes

def nms(boxes, conf_thresh=0.2, iou_thresh=0.5):
    boxes = [b for b in boxes if b[1] > conf_thresh]
    boxes.sort(key=lambda x: x[1], reverse=True)
    result = []
    
    while boxes:
        chosen = boxes.pop(0)
        boxes = [b for b in boxes if b[0] != chosen[0] or iou(chosen[2:], b[2:]) < iou_thresh]
        result.append(chosen)
    
    return result

def iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    union = (x2_1-x1_1)*(y2_1-y1_1) + (x2_2-x1_2)*(y2_2-y1_2) - inter + 1e-6
    
    return inter / union

def draw_boxes(img, boxes):
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    for box in boxes:
        clss, conf, x, y, bw, bh = box
        x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
        x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)

        cv2.rectangle(img_array, (x1, y1), (x2, y2), colors[int(clss)], 2)

        label = f"{CLASS_NAMES[int(clss)]}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(img_array, 
                    (x1, y1 - text_h - baseline - 5),  
                    (x1 + text_w, y1),                  
                    colors[int(clss)],                   
                    cv2.FILLED)                         
        cv2.putText(img_array, label, (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return Image.fromarray(img_array)


# Load model
model = Yolov1()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        img_resized = img.resize((448, 448))
        img_tensor = torch.from_numpy(np.array(img_resized) / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img_tensor)
        
        boxes = cellboxes_to_boxes(pred)[0].numpy()
        boxes_list = [boxes[i,j].tolist() for i in range(14) for j in range(14) if boxes[i,j,1] > 0.25]
        boxes_nms = nms(boxes_list)
        
        detections = [{'class_name': CLASS_NAMES[int(b[0])], 'confidence': float(b[1]), 'bin': BINS[int(b[0])]} for b in boxes_nms]
        
        img_with_boxes = draw_boxes(img, boxes_nms)
        buffered = io.BytesIO()
        img_with_boxes.save(buffered, format="JPEG")
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image': f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)


