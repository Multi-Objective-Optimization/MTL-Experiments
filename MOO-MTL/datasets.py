import torch
from torchvision import transforms
from loaders.multi_mnist_loader import MNIST
from loaders.cityscapes_loader import CITYSCAPES
from loaders.augmentations import Compose, RandomRotate, RandomHorizontallyFlip
from loaders.celeba_loader import CELEBA


def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(cfg):
    if 'dataset' not in cfg:
        print('ERROR: No dataset is specified')

    num_workers = cfg['training']['num_workers']
    batch_size = cfg['batch_size']
    data_root = cfg['data_root']

    if 'mnist' in cfg['dataset']:
        batch_size_val = cfg.get('batch_size_val') or batch_size
        train_dst = MNIST(root=data_root, train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        val_dst = MNIST(root=data_root, train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size_val, shuffle=True, num_workers=num_workers)
        return train_loader, train_dst, val_loader, val_dst

    if 'cityscapes' in cfg['dataset']:
        img_size = tuple(cfg['image_size'])
        depth_mean_path = cfg['depth_mean_path']
        augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])

        train_dst = CITYSCAPES(root=data_root, is_transform=True, split=['train'],
                               img_size=img_size, augmentations=augmentations,
                               depth_mean_path=depth_mean_path)
        val_dst = CITYSCAPES(root=data_root, is_transform=True, split=['val'],
                             img_size=img_size, depth_mean_path=depth_mean_path)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, num_workers=num_workers)
        return train_loader, train_dst, val_loader, val_dst

    if 'celeba' in cfg['dataset']:
        img_size = tuple(cfg['image_size'])

        train_dst = CELEBA(root=data_root, is_transform=True, split='train', img_size=img_size, augmentations=None)
        val_dst = CELEBA(root=data_root, is_transform=True, split='val', img_size=img_size, augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, num_workers=num_workers)
        return train_loader, train_dst, val_loader, val_dst
