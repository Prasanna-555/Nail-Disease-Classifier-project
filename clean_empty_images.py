import os

def delete_empty_images(folder_path):
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.getsize(filepath) == 0:
                print(f"Deleting empty file: {filepath}")
                os.remove(filepath)
                count += 1
    print(f"✅ Removed {count} empty files.")

# Run this on both train and valid
delete_empty_images("dataset/train")
delete_empty_images("dataset/valid")
