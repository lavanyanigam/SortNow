# NOTE: These preprocessing steps were done on the original datset availible at Roboflow: https://universe.roboflow.com/material-identification/garbage-classification-3
# NOTE: My dataset after running the preprocessing is availible on Kaggle : https://www.kaggle.com/datasets/lavanyanigam/garbageclassificationfinal 


import os

def filter_excessive_annotations(images_path, labels_path, max_objects=98):
    removed_count = 0
    kept_count = 0
    
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(labels_path, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_path, image_name)
        
        with open(label_path) as f:
            lines = f.readlines()
        
        if len(lines) > max_objects:
            if os.path.exists(label_path):
                os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            removed_count += 1
            print(f"Removed: {image_name} ({len(lines)} objects)")
        else:
            kept_count += 1
    
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Images kept (≤{max_objects} objects): {kept_count}")
    print(f"Images removed (>{max_objects} objects): {removed_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter images with excessive annotations')
    parser.add_argument('--images', required=True, help='Path to images folder')
    parser.add_argument('--labels', required=True, help='Path to labels folder')
    parser.add_argument('--max', type=int, default=98, help='Max objects per image')
    
    args = parser.parse_args()
    filter_excessive_annotations(args.images, args.labels, args.max)
