import os
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from torchvision import transforms as T  
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-4
EPOCHS = 100
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "checkpoint.pth"

CLASS_NAMES = {
    0: 'BIODEGRADABLE',
    1: 'CARDBOARD',
    2: 'GLASS',
    3: 'METAL',
    4: 'PAPER',
    5: 'PLASTIC'
}

IMG_DIR = "/kaggle/input/garbageclassificationfinal/GARBAGECLASSIFICATIONFINAL/train/images"
LABEL_DIR = "/kaggle/input/garbageclassificationfinal/GARBAGECLASSIFICATIONFINAL/train/labels"
TEST_IMG_DIR = "/kaggle/input/garbageclassificationfinal/GARBAGECLASSIFICATIONFINAL/test/images"
TEST_LABEL_DIR = "/kaggle/input/garbageclassificationfinal/GARBAGECLASSIFICATIONFINAL/test/labels"
BASE_OUTPUT = "/kaggle/working/"
MODEL_PATH = os.path.join(BASE_OUTPUT, "models", "objdetector.pth")
PLOTS_PATH = os.path.join(BASE_OUTPUT, "plots")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

print(f"device: {DEVICE}")
print(f"Output directory: {BASE_OUTPUT}")
# CLASS WEIGHTS
train_object_counts = [21971, 3372, 5196, 3828, 2981, 3866]
total_objects = 41214

class_weights_raw = [total_objects / count for count in train_object_counts]
max_weight = 13.824

class_weights_normalized = [w / max_weight for w in class_weights_raw]
class_weights = torch.tensor(class_weights_normalized, dtype=torch.float32).to(DEVICE)

print("CLASS WEIGHTS:")
labels = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
for label, weight in zip(labels, class_weights_normalized):
    print(f"  {label}: {weight:.3f}")
print("="*50 + "\n")


# device: cuda
# Output directory: /kaggle/working/
# CLASS WEIGHTS:
#   biodegradable: 0.136
#   cardboard: 0.884
#   glass: 0.574
#   metal: 0.779
#   paper: 1.000
#   plastic: 0.771


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == "midpoint":
        b1x1 = boxes_preds[...,0:1]-boxes_preds[...,2:3]/2
        b1y1 = boxes_preds[...,1:2]-boxes_preds[...,3:4]/2
        b1x2 = boxes_preds[...,0:1]+boxes_preds[...,2:3]/2
        b1y2 = boxes_preds[...,1:2]+boxes_preds[...,3:4]/2
        
        b2x1 = boxes_labels[...,0:1]-boxes_labels[...,2:3]/2
        b2y1 = boxes_labels[...,1:2]-boxes_labels[...,3:4]/2
        b2x2 = boxes_labels[...,0:1]+boxes_labels[...,2:3]/2
        b2y2 = boxes_labels[...,1:2]+boxes_labels[...,3:4]/2

    if box_format == "corners":
        b1x1 = boxes_preds[...,0:1]
        b1y1 = boxes_preds[...,1:2]
        b1x2 = boxes_preds[...,2:3]
        b1y2 = boxes_preds[...,3:4]
        
        b2x1 = boxes_labels[...,0:1]
        b2y1 = boxes_labels[...,1:2]
        b2x2 = boxes_labels[...,2:3]
        b2y2 = boxes_labels[...,3:4]

    x1 = torch.max(b1x1, b2x1)
    y1 = torch.max(b1y1, b2y1)
    x2 = torch.min(b1x2, b2x2)
    y2 = torch.min(b1y2, b2y2)

    inter = (x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    b1_area = abs((b1x2-b1x1)*(b1y2-b1y1))
    b2_area = abs((b2x2-b2x1)*(b2y2-b2y1))
    
    union = b1_area+b2_area-inter+1e-6

    return inter/union
class CustomTensorDataset(Dataset):
    def __init__(self, img_dir, label_dir, S=14, B=2, C=6, transform=None):
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = []
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        for img_path in all_imgs:
            label_path = self._get_label_path(img_path)
            if os.path.exists(label_path):
                self.img_paths.append(img_path)
        
        print(f"Number of images loaded: {len(self.img_paths)}")

    def _get_label_path(self, img_path):
        basename = os.path.basename(img_path).replace('.jpg', '.txt')
        return os.path.join(self.label_dir, basename)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = self._get_label_path(img_path)
        boxes = []
        class_labels = []  
        
        with open(label_path) as f:
            for line in f.readlines():
                line = line.strip().split()
                if len(line) < 5:
                    continue
                class_label = int(line[0])
                x_center, y_center, width, height = map(float, line[1:5])
                if class_label >= self.C:
                    print(f"warning: class {class_label} >= {self.C} in {label_path}")
                    continue
                # Store bounding box (without class) and class separately
                boxes.append([x_center, y_center, width, height])
                class_labels.append(class_label)
    
        if self.transform is not None and len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
            
        image = cv2.resize(image, (448, 448))
        image = T.ToTensor()(image)
    
        # Target format: [C classes, confidence, x, y, w, h, (unused B=2 slots)]
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        
        # Reconstruct boxes with their class labels
        for box, class_label in zip(boxes, class_labels):
            if len(box) != 4:  # Skip invalid boxes
                continue
                
            x_center, y_center, width, height = box
            class_label = int(class_label)
            
            # Find grid cell
            i = int(self.S * y_center)  
            j = int(self.S * x_center)  
            i = max(0, min(i, self.S - 1))  # Clamp to valid range
            j = max(0, min(j, self.S - 1))
    
            # Only store ONE box per cell (first one encountered)
            if label_matrix[i, j, 6] == 0:
                # Calculate cell-relative coordinates
                x_cell = self.S * x_center - j  
                y_cell = self.S * y_center - i  
                width_cell = width * self.S
                height_cell = height * self.S
    
                # Store in first box slot only
                label_matrix[i, j, class_label] = 1  # One-hot class
                label_matrix[i, j, 6] = 1  # Confidence (objectness)
                label_matrix[i, j, 7:11] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                # Indices 11:16 remain zeros (unused for ground truth)
        
        return image, label_matrix
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
        self.conv= nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm=nn.BatchNorm2d(out_channels)
        self.leakyrelu= nn.LeakyReLU(0.1,inplace=False)
    
    def forward(self, x):
        x= self.conv(x)
        x=self.batchnorm(x)
        x=self.leakyrelu(x)
        return x
        
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs_(**kwargs)

    def forward(self, x):
        x=self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))

    def _create_conv_layers(self,architecture):
        layers=[]
        in_channels=self.in_channels

        for x in architecture:
            if type(x)==tuple:
                layers+=[
                    CNNBlock(
                        in_channels, 
                        out_channels=x[1], 
                        kernel_size=x[0], 
                        stride=x[2], 
                        padding=x[3],
                    )
                ]
                in_channels=x[1]
                
            elif type(x)==str:
                layers+=[ nn.MaxPool2d(kernel_size=2,stride=2)]

              
        return nn.Sequential(*layers)
        
    def _create_fcs_(self, split_size, num_boxes, num_classes):
            S,B,C=split_size,num_boxes,num_classes
            return nn.Sequential(
                nn.Linear(512*S*S, 2048),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(2048, S*S*(C+B*5)),
            )
class YoloLoss(nn.Module):
    def __init__(self, S=14, B=2, C=6, class_weights=None):  # ADD class_weights parameter
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.class_weights = class_weights  #  store weights

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        target = target.reshape(-1, self.S, self.S, self.C + self.B*5)

        # IoUs:  [batch, S, S]
        iou_b1 = intersection_over_union(predictions[..., 7:11], target[..., 7:11])
        iou_b2 = intersection_over_union(predictions[..., 12:16], target[..., 7:11])
    
        # If IoU returns [batch, S, S, 1], squeeze last dim
        if iou_b1.dim() == 4 and iou_b1.size(-1) == 1:
            iou_b1 = iou_b1.squeeze(-1)   # [batch, S, S]
            iou_b2 = iou_b2.squeeze(-1)   # [batch, S, S]
    
        # stack [batch, S, S, 2]
        ious = torch.stack([iou_b1, iou_b2], dim=-1)
    
        # Max over last dim best box index in {0,1}, shape [batch, S, S]
        iou_maxes, bestbox = torch.max(ious, dim=-1)
    
        exists_box = target[..., 6].unsqueeze(-1)  # [batch, S, S, 1]
    
        # Make bestbox broadcastable: [batch, S, S, 1]
        bestbox = bestbox.unsqueeze(-1).float()
    
        # Box coordinates
        box_predictions = exists_box * (
            bestbox * predictions[..., 12:16] +
            (1 - bestbox) * predictions[..., 7:11]
        )
        box_targets = exists_box * target[..., 7:11]

        # transform w, h without modifying in-place
        box_predictions = torch.cat([
            box_predictions[..., :2],
            torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
                torch.abs(box_predictions[..., 2:4]) + 1e-6
            )
        ], dim=-1)
        
        box_targets = torch.cat([
            box_targets[..., :2],
            torch.sqrt(box_targets[..., 2:4])
        ], dim=-1)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2), 
        )

        # obj loss
        pred_box = bestbox * predictions[..., 11:12] + (1 - bestbox) * predictions[..., 6:7]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 6:7])
        )

        # no obj loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 6:7], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 6:7], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 11:12], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 6:7], start_dim=1)
        )

        
        if self.class_weights is not None:
            # Get target class index for each cell [batch, S, S]
            target_class_idx = torch.argmax(target[..., :6], dim=-1)
            
            # Get weight for each cell based on its class [batch, S, S]
            weights = self.class_weights[target_class_idx]
            
            # Make weights broadcastable to class dimension [batch, S, S, 1]
            weights = weights.unsqueeze(-1)
            
            
            weighted_pred = exists_box * weights * predictions[..., :6]
            weighted_target = exists_box * weights * target[..., :6]
            
            class_loss = self.mse(
                torch.flatten(weighted_pred, end_dim=-2),
                torch.flatten(weighted_target, end_dim=-2)
            )
        else:
            
            class_loss = self.mse(
                torch.flatten(exists_box * predictions[..., :6], end_dim=-2),
                torch.flatten(exists_box * target[..., :6], end_dim=-2)
            )

        loss = (
            self.lambda_coord * box_loss + 
            object_loss + 
            self.lambda_noobj * no_object_loss + 
            class_loss
        )

        return loss
def train_fn(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (input_tensor, target_tensor) in enumerate(loop):
        input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)
        out = model(input_tensor)
        loss = loss_fn(out, target_tensor)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clear memory
        del out
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(losses)
    return mean_loss


def validate_fn(val_loader,model, loss_fn):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for input_tensor, target_tensor in val_loader:
            input_tensor, target_tensor = input_tensor.to(DEVICE), target_tensor.to(DEVICE)
            out = model(input_tensor)
            loss = loss_fn(out, target_tensor)
            losses.append(loss.item())
    
    mean_loss = sum(losses)/len(losses)
    return mean_loss

def save_checkpoint(state, filename="checkpoint.pth"):
    print(f"=[INFO]Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("[INFO]Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def plot_train_hist(train_losses, val_losses, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 1, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()
    plot_file = os.path.join(save_path, 'training_history.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Training plot saved to {plot_file}")
    plt.show()


def main():
    model = Yolov1(split_size=14, num_boxes=2, num_classes=6).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=14, B=2, C=6, class_weights=class_weights)

    checkpoint_path = os.path.join(BASE_OUTPUT, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_checkpoint(checkpoint, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])     
        val_losses = checkpoint.get('val_losses', [])         
        best_epoch = start_epoch                               
        print(f"[INFO] ✓ Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        best_epoch = 0                                         
        print("[INFO] Starting fresh training...")

    
    train_dataset = CustomTensorDataset(
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=14, B=2, C=6,
    )

    test_dataset = CustomTensorDataset(
        img_dir=TEST_IMG_DIR,
        label_dir=TEST_LABEL_DIR,
        S=14, B=2, C=6,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=True,
    )
    
    # Early stopping
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    patience_counter = 0
    patience_limit = 10
    print ("="*50)
    print(f"Model Linear layer input size: {model.fcs[0].in_features}")
    print(f"Expected: {512*14*14}")
    print ("="*50)
    for epoch in range(start_epoch, EPOCHS):  
        print(f"\n[INFO] EPOCH {epoch+1}/{EPOCHS}")

        # Train and validate
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        train_losses.append(train_loss)
        print(f"[INFO] Training Loss: {train_loss:.4f}")

        val_loss = validate_fn(test_loader, model, loss_fn)
        val_losses.append(val_loss)
        print(f"[INFO] Validation Loss: {val_loss:.4f}")

        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model and early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # Reset counter
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_losses": train_losses,   
                "val_losses": val_losses,       
            }

            save_checkpoint(checkpoint, filename=os.path.join(BASE_OUTPUT, "best_model.pth"))
            print(f"[INFO] New best model saved! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{patience_limit}")
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            break


    torch.save(model.state_dict(), os.path.join(BASE_OUTPUT, "final_garbage_detector.pth"))

    plot_train_hist(train_losses, val_losses, PLOTS_PATH)
    
    print("\n" + "="*50)
    print("[INFO] TRAINING COMPLETE")
    print(f"[INFO] Best model at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    print(f"[INFO] Final training loss: {train_losses[-1]:.4f}")
    print(f"[INFO] Final validation loss: {val_losses[-1]:.4f}")
    print(f"[INFO] Total epochs trained: {len(train_losses)}")
    print("="*50)

if __name__ == "__main__":
    main()

# Number of images loaded: 7260
# Number of images loaded: 3114
# Model Linear layer input size: 100352

# [INFO] Early stopping triggered at epoch 64
# [INFO] Training plot saved to /kaggle/working/plots/training_history.png
# [INFO] TRAINING COMPLETE
# [INFO] Best model at epoch 54 with validation loss: 176.3113
# [INFO] Final training loss: 65.5796
# [INFO] Final validation loss: 177.0820
# [INFO] Total epochs trained: 64
