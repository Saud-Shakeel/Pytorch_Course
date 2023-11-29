import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def data_loaders(train_dir: str,
                test_dir: str,
                batch_size: int,
                data_transform: transforms.Compose,
                num_os_workers: os.cpu_count()):
    
    train_dataset = datasets.ImageFolder(root = train_dir,
                                        transform = data_transform,
                                        target_transform= None)
    
    test_dataset = datasets.ImageFolder(root = test_dir,
                                        transform = data_transform,
                                        target_transform = None)
    
    num_classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    train_dataLoader = DataLoader(dataset = train_dataset,
                                shuffle= True,
                                batch_size= batch_size,
                                num_workers= num_os_workers)
    
    
    test_dataLoader = DataLoader(dataset = test_dataset,
                                shuffle= False,
                                batch_size= batch_size,
                                num_workers= num_os_workers)
    
    return train_dataLoader, test_dataLoader, num_classes, class_to_idx

