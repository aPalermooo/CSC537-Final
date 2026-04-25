import cv2
from torch.utils.data import Dataset

class CharacterDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.subset[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.subset)