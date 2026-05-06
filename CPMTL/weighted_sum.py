import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from utils.config import load_config, parse_args
from utils.metrics import evaluate
from utils.misc import evenly_dist_weights
from dataset import MultiMNIST
from model import MultiLeNet


def train(pref, ckpt_name, cfg):
    seed = cfg['train']['seed']
    batch_size = cfg['data']['batch_size']
    num_workers = cfg['data']['num_workers']
    lr = cfg['train']['lr']
    momentum = cfg['train']['momentum']
    weight_decay = cfg['train']['weight_decay']
    num_epochs = cfg['train']['epochs']

    dataset_path = Path(cfg['data']['root'])
    ckpt_path = Path(cfg['output']['dir'])

    dataset_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MultiMNIST(dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = MultiMNIST(dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    network = MultiLeNet().to(device)

    criterion = F.cross_entropy
    closures = [lambda n, l, t: criterion(l[0], t[:, 0]), lambda n, l, t: criterion(l[1], t[:, 1])]

    optimizer = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, num_epochs * len(trainloader))

    random_ckpt_path = ckpt_path / 'random.pth'
    if not random_ckpt_path.is_file():
        torch.save({
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, random_ckpt_path)
    random_ckpt = torch.load(random_ckpt_path, map_location='cpu', weights_only=False)
    network.load_state_dict(random_ckpt['state_dict'])
    optimizer.load_state_dict(random_ckpt['optimizer'])
    lr_scheduler.load_state_dict(random_ckpt['lr_scheduler'])

    evaluate(network, testloader, device, closures, f'{ckpt_name}')

    num_steps = len(trainloader)
    for epoch in range(1, num_epochs + 1):
        network.train(True)
        trainiter = iter(trainloader)
        for _ in range(1, num_steps + 1):
            images, labels = next(trainiter)
            images = images.to(device)
            labels = labels.to(device)
            logits = network(images)
            losses = [c(network, logits, labels) for c in closures]
            loss = sum(w * l for w, l in zip(pref, losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}: {epoch}/{num_epochs}')

    torch.save({
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'preference': pref,
        'record': {'losses': losses, 'tops': tops},
    }, ckpt_path / f'{ckpt_name}.pth')


def main(cfg):
    n_prefs = cfg['train']['n_prefs']
    prefs = evenly_dist_weights(n_prefs + 2, 2)
    for i, pref in enumerate(prefs):
        train(pref, str(i), cfg)


if __name__ == '__main__':
    args = parse_args('configs/weighted_sum.yaml')
    cfg = load_config(args.config, args.overrides)
    main(cfg)
