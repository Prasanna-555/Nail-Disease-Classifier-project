import os

root_dir = "dataset/train"  # or "dataset/valid"
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

for class_name in os.listdir(root_dir):
    folder = os.path.join(root_dir, class_name)
    if os.path.isdir(folder):
        files = os.listdir(folder)
        valid_images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions and os.path.getsize(os.path.join(folder, f)) > 0]
        print(f"{class_name}: {len(valid_images)} valid images")
