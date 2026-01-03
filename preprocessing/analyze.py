# NOTE: These preprocessing steps were done on the original datset availible at Roboflow: https://universe.roboflow.com/material-identification/garbage-classification-3
# NOTE: My dataset after running the preprocessing is availible on Kaggle : https://www.kaggle.com/datasets/lavanyanigam/garbageclassificationfinal 


import os
import cv2
import random

def analyze_dataset(images_path, labels_path):
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg','.png','.jpeg'))]
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    print(f"Total images: {len(image_files)}")
    print(f"Total labels: {len(label_files)}\n")
    
    missing_labels = []
    corrupted_images = []
    label_issues = []
    
    for img_name in image_files:
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        if label_name not in label_files:
            missing_labels.append(img_name)
            continue
     
        img_path = os.path.join(images_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            corrupted_images.append(img_name)
            continue
        
        label_path = os.path.join(labels_path, label_name)
        try:
            with open(label_path) as f:
                lines = f.readlines()
                if len(lines) == 0:
                    label_issues.append((img_name, "empty label file"))
        except:
            label_issues.append((img_name, "cannot read"))
    
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Label issues: {len(label_issues)}\n")
    
    class_counts = {}
    total_objects = 0
    images_per_class = {str(i): 0 for i in range(6)}
    
    for label_file in label_files:
        label_path = os.path.join(labels_path, label_file)
        try:
            with open(label_path) as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
                
                classes_in_image = set([line.split()[0] for line in lines if len(line.split()) >= 5])
                for c in classes_in_image:
                    images_per_class[c] = images_per_class.get(c, 0) + 1
        except:
            pass
    
    print("=== CLASS DISTRIBUTION (Total Objects) ===")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        print(f"Class {class_id}: {count} objects ({percentage:.1f}%)")
    
    print(f"\nTotal objects: {total_objects}")
    print(f"\n=== IMAGES PER CLASS ===")
    for class_id in sorted(images_per_class.keys()):
        print(f"Class {class_id}: {images_per_class[class_id]} images")
    
    print("\n=== SAMPLE ANNOTATIONS ===")
    sample_labels = random.sample(label_files, min(5, len(label_files)))
    for label_file in sample_labels:
        label_path = os.path.join(labels_path, label_file)
        with open(label_path) as f:
            lines = f.readlines()
            print(f"{label_file}: {len(lines)} objects")
            if lines:
                print(f"  First annotation: {lines[0].strip()}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze YOLO dataset')
    parser.add_argument('--images', required=True, help='Path to images folder')
    parser.add_argument('--labels', required=True, help='Path to labels folder')
    
    args = parser.parse_args()
    analyze_dataset(args.images, args.labels)
