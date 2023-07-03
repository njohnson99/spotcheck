import numpy as np
from PIL import Image
import sys
import torch
import json
from torchvision import transforms
from torchvision.datasets import VisionDataset

def get_transform(mode = 'normalize'):
    if mode == 'normalize':
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif mode == 'reshape':
        return transforms.Compose([
                MakeSquare(),
                transforms.Resize((224,224))
                ])
    elif mode == 'full':
        return transforms.Compose([
                MakeSquare(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    elif mode == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif mode == 'resize-crop':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            ])
    
def load_data(ids, images, data_key = 'file', index = None):
    files = []
    labels = []
    for i in ids:
        files.append(images[i][data_key])
        if index is None:
            labels.append(images[i]['label'])
        else:
            labels.append(images[i]['label'][index])

    labels = np.array(labels, dtype = np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)
        
    return files, labels

def load_phase(source, phase, index = None):
    with open('{}/{}/images.json'.format(source, phase), 'r') as f:
        images = json.load(f)
    ids = list(images)
    
    files, labels = load_data(ids, images, index = index)
    
    return files, labels

def get_loader(dataset, batch_size = 64, num_workers = 0):
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)

class ImageDataset(VisionDataset):
    
    def __init__(self, filenames, labels, transform_mode = 'normalize', get_names = False,):
        transform = get_transform(mode = transform_mode)
        super(ImageDataset, self).__init__(None, None, transform, None)
        self.filenames = filenames
        self.labels = labels
        self.get_names = get_names
        
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        if self.get_names:
            return img, label, filename
        else:
            return img, label
        
    def __len__(self):
        return len(self.filenames)