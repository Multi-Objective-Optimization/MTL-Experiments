import os
import torch
import numpy as np
from PIL import Image
import re
import glob

from torch.utils import data


class CELEBA(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=(32, 32), augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 40
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}
        self.labels = {}

        self.label_file = os.path.join(self.root, 'Anno', 'list_attr_celeba.txt')
        label_map = {}
        with open(self.label_file, 'r') as l_file:
            labels = l_file.read().split('\n')[2:-1]
        for label_line in labels:
            f_name = re.sub('jpg', 'png', label_line.split(' ')[0])
            label_txt = list(map(lambda x:int(x), re.sub('-1','0',label_line).split()[1:]))
            label_map[f_name] = label_txt

        self.all_files = glob.glob(os.path.join(self.root, 'Img', 'img_align_celeba_png', '*.png'))
        with open(os.path.join(self.root, 'Eval', 'list_eval_partition.txt'), 'r') as f:
            fl = f.read().split('\n')
            fl.pop()
            if 'train' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='0', fl))
            elif 'val' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='1', fl))
            elif 'test' in self.split:
                selected_files = list(filter(lambda x:x.split(' ')[1]=='2', fl))
            selected_file_names = list(map(lambda x:re.sub('jpg', 'png', x.split(' ')[0]), selected_files))

        if len(self.all_files) == 0:
            raise Exception("No CelebA images found in %s" % self.root)

        base_path = os.path.dirname(self.all_files[0])
        available_files = set(map(lambda x:x.split('/')[-1], self.all_files))
        split_file_names = sorted(available_files.intersection(set(selected_file_names)))
        self.files[self.split] = [os.path.join(base_path, x) for x in split_file_names]
        self.labels[self.split] = [label_map[x] for x in split_file_names]
        self.class_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
                                'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        if len(self.files[self.split]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))

        print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        label = self.labels[self.split][index]
        img = np.array(Image.open(img_path))

        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))

        if self.is_transform:
            img = self.transform_img(img)

        return [img] + label

    def transform_img(self, img):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = np.array(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((self.img_size[1], self.img_size[0]), Image.BILINEAR))
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img
