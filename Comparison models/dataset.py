import random
from pathlib import Path
from typing import List, Sequence, Dict, Tuple, Optional
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Clsdata(Dataset):
    """
    root_list: 
    train / test: 
    """
    def __init__(
        self,
        root_list: Sequence[str],
        shuffle: bool = True,
        transform=None,
        train: bool = False,
        test: bool = False,
        img_size: int = 224,
    ):
        assert not (train and test), 
        self.test = test
        self.train = train
        self.img_size = int(img_size)

        lines: List[Path] = [Path(p) for p in root_list]

        if shuffle:
            random.shuffle(lines)

        self.lines = lines
        self.nSamples = len(self.lines)

        if transform is None:
            letterbox = LetterboxToSquare(final_size=self.img_size, fill=pad_fill)
            if self.train and not self.test:
                self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
                ])
            else:
                # val / test
                self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225)),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return self.nSamples


    def __getitem__(self, index: int):
        assert 0 <= index < len(self), f'index out of range: {index} / {len(self)}'

        img_path = self.lines[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if 'Proximal ulna and radius fractures' in str(img_path):
            label = 0
        else:
            if 'Midshaft ulna and radius fractures' in str(img_path):
                label = 1
            else:
                label = 2


        return img, label

