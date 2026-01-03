# NOTE: These preprocessing steps were done on the original datset availible at Roboflow: https://universe.roboflow.com/material-identification/garbage-classification-3
# NOTE: My dataset after running the preprocessing is availible on Kaggle : https://www.kaggle.com/datasets/lavanyanigam/garbageclassificationfinal 

import os
import shutil

def filter_single_object_images(images_path, labels_path, output_images, output_labels):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    
    class_counts = [0, 0, 0, 0, 0, 0]
    total_kept = 0
    total_removed = 0
    
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_path, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_path, image_name)
        
        if not os.path.exists(image_path):
            continue
        
        with open(label_path) as f:
            lines = f.readlines()
        
        if len(lines) == 1:
            shutil.copy(image_path, os.path.join(output_images, image_name))
            shutil.copy(label_path, os.path.join(output_labels, label_file))
            
            class_id = int(lines[0].strip().split()[0])
            if 0 <= class_id < 6:
                class_counts[class_id] += 1
            
            total_kept += 1
        else:
            total_removed += 1
    
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Images kept (single obj): {total_kept}")
    print(f"Images removed (multiple obj): {total_removed}")
    print(f"\nNEW CLASS DISTRIBUTION  ")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {count} images")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter single-object images')
    parser.add_argument('--images', required=True, help='Source images folder')
    parser.add_argument('--labels', required=True, help='Source labels folder')
    parser.add_argument('--out-images', required=True, help='Output images folder')
    parser.add_argument('--out-labels', required=True, help='Output labels folder')
    
    args = parser.parse_args()
    filter_single_object_images(args.images, args.labels, args.out_images, args.out_labels)
