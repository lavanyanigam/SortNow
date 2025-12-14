#cell 1 

!pip install --upgrade matplotlib opencv-python
!pip install ultralytics "numpy<2"
import numpy as np
import torch
print(torch.cuda.is_available())  
print(torch.cuda.get_device_name(0))  
print(f"NumPy Version: {np.__version__}")

#cell 2

import yaml
import os

data_config = {
'path': '/kaggle/input/waste-dataset',
'train': 'train/images',
'val': 'valid/images',
'names': {
    0: 'Electronics',
    1: 'biological',
    2: 'cardboard',
    3: 'clothes',
    4: 'glass',
    5: 'metal',
    6: 'paper',
    7: 'plastic',
    8: 'shoes'
}
}
with open('/kaggle/working/data.yaml', 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)
print("data.yaml updated")


#cell 3

from ultralytics import YOLO

model = YOLO('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt')
results = model.train(
    data='/kaggle/working/data.yaml',
    imgsz=416,
    augment=False,
    epochs=50,
    batch=16,
    patience=10,
    save_period=5,
    device=0,
    verbose=True,
    exist_ok=True,
    cache=False,
    amp=False,
    project='/kaggle/working',
    name='sortnow_model',
    save=True
)

#cell 4

import os
import shutil
from IPython.display import FileLink

metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

os.makedirs('/kaggle/working/model_weights', exist_ok=True)
shutil.copy('/kaggle/working/runs/detect/train/weights/best.pt', 
            '/kaggle/working/model_weights/best.pt')
shutil.copy('/kaggle/working/runs/detect/train/weights/last.pt', 
            '/kaggle/working/model_weights/last.pt')
print("training finished")
display(FileLink('/kaggle/working/model_weights/best.pt'))
display(FileLink('/kaggle/working/model_weights/last.pt'))
