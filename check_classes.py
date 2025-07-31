from torchvision import datasets
import os

data_dir = 'dataset/train'  # Make sure this path is correct

dataset = datasets.ImageFolder(data_dir)
print("Classes:", dataset.classes)
print("Number of classes:", len(dataset.classes))
