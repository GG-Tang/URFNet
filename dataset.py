import random
from pathlib import Path
from typing import List, Sequence, Optional
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class Clsdata(Dataset):
    """
    root_list: 传入图片绝对路径列表  (List[str] or List[Path])
    train / test: 二选一控制增广与评估变换（都为 False 时按验证/评估处理）
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
        assert not (train and test), "train 与 test 不能同时为 True"
        self.test = test
        self.train = train
        self.img_size = int(img_size)

        # 复制一份，避免原列表被打乱
        lines: List[Path] = [Path(p) for p in root_list]
        if shuffle:
            random.shuffle(lines)
        self.lines = lines
        self.nSamples = len(self.lines)

        # 统一的均值方差
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)

        if transform is None:
            if self.train and not self.test:
                # 训练阶段我们用“成对变换”，故这里不再放随机增广的 Compose
                self.transform = None
            else:
                # val / test：确定性单图变换
                self.transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return self.nSamples

    # --- 成对随机增广：保证两张图共享同一随机性 ---
    def _pair_transform(self, img1_pil: Image.Image, img2_pil: Image.Image):
        # 保护 EXIF 方向
        img1 = ImageOps.exif_transpose(img1_pil)
        img2 = ImageOps.exif_transpose(img2_pil)

        # 统一 Resize 到 224（与原逻辑一致；如需 Letterbox 可在此改）
        img1 = TF.resize(img1, [self.img_size, self.img_size])
        img2 = TF.resize(img2, [self.img_size, self.img_size])

        # 同步随机水平翻转
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # 同步随机垂直翻转
        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # 同步颜色抖动（手动采样参数，保证两张一致）
        # 对齐你原来的范围：brightness/contrast/saturation=0.2, hue=0.05
        if random.random() < 0.8:  # 可选：控制是否进行颜色增广
            b = 1.0 + random.uniform(-0.2, 0.2)  # brightness factor
            c = 1.0 + random.uniform(-0.2, 0.2)  # contrast factor
            s = 1.0 + random.uniform(-0.2, 0.2)  # saturation factor
            h = random.uniform(-0.05, 0.05)      # hue delta

            img1 = TF.adjust_brightness(img1, b); img2 = TF.adjust_brightness(img2, b)
            img1 = TF.adjust_contrast(img1, c);   img2 = TF.adjust_contrast(img2, c)
            img1 = TF.adjust_saturation(img1, s); img2 = TF.adjust_saturation(img2, s)
            img1 = TF.adjust_hue(img1, h);        img2 = TF.adjust_hue(img2, h)

        # ToTensor + Normalize
        img1 = TF.to_tensor(img1); img2 = TF.to_tensor(img2)
        img1 = TF.normalize(img1, self.mean, self.std)
        img2 = TF.normalize(img2, self.mean, self.std)
        return img1, img2

    def __getitem__(self, index: int):
        assert 0 <= index < len(self), f'index out of range: {index} / {len(self)}'

        img_path = self.lines[index]
        # front / side
        img1_path = img_path
        img2_path = Path(str(img_path).replace("front.jpg", "side.jpg"))

        # 读成 PIL
        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')

        if self.train and not self.test and self.transform is None:
            # 训练：两张图共享同一随机增广
            img, img2 = self._pair_transform(img1_pil, img2_pil)
        else:
            # 验证/测试：确定性单图变换
            img  = self.transform(img1_pil)
            img2 = self.transform(img2_pil)

        # if-else 打标签（保持你的写法）
        p = str(img_path)
        if 'Proximal ulna and radius fractures' in p:
            label = 0
        else:
            if 'Midshaft ulna and radius fractures' in p:
                label = 1
            else:
                label = 2

        return img, img2, label



