import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image


class LicenseDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.transform = transform
        self.img_labels = pd.read_csv(os.path.join(img_dir, '_annotations.csv'))
        self.img_dir = img_dir
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(str(img_path))
        label = self.img_labels.iloc[idx, 1]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label