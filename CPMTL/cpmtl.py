import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms

from utils.config import load_config, parse_args
from utils.metrics import evaluate
from utils.misc import TopTrace
from solvers.hvp_solver import VisionHVPSolver
from solvers.kkt_solver import MINRESKKTSolver
from dataset import MultiMNIST
from model import MultiLeNet


def train(start_path, beta, cfg):
    seed = cfg['train']['seed']
    batch_size = cfg['data']['batch_size']
    num_workers = cfg['data']['num_workers']
    lr = cfg['train']['lr']
    momentum = cfg['train']['momentum']
    weight_decay = cfg['train']['weight_decay']
    num_steps = cfg['train']['n_steps']

    damping = cfg['solver']['damping']
    maxiter = cfg['solver']['maxiter']
    tol = cfg['solver']['tolerance']
    shift = cfg['solver']['shift']
    kkt_momentum = cfg['solver']['kkt_momentum']
    stochastic = cfg['solver']['stochastic']

    ckpt_name = start_path.name.split('.')[0]
    dataset_path = Path(cfg['data']['root'])
    ckpt_path = Path(cfg['output']['dir']) / ckpt_name

    if not start_path.is_file():
        raise RuntimeError('Pareto solutions not found.')

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

    start_ckpt = torch.load(start_path, map_location='cpu', weights_only=False)
    network.load_state_dict(start_ckpt['state_dict'])

    criterion = F.cross_entropy
    closures = [lambda n, l, t: criterion(l[0], t[:, 0]), lambda n, l, t: criterion(l[1], t[:, 1])]

    hvp_solver = VisionHVPSolver(network, device, trainloader, closures, shared=False)
    hvp_solver.set_grad(batch=False)
    hvp_solver.set_hess(batch=True)

    kkt_solver = MINRESKKTSolver(
        network, hvp_solver, device,
        stochastic=stochastic, kkt_momentum=kkt_momentum, create_graph=False,
        grad_correction=False, shift=shift, tol=tol, damping=damping, maxiter=maxiter)

    optimizer = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}')

    top_trace = TopTrace(len(closures))
    top_trace.print(tops, show=False)

    beta = beta.to(device)

    for step in range(1, num_steps + 1):
        network.train(True)
        optimizer.zero_grad()
        kkt_solver.backward(beta, verbose=False)
        optimizer.step()

        losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}: {step}/{num_steps}')
        top_trace.print(tops)

        torch.save({
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'beta': beta,
            'record': {'losses': losses, 'tops': tops},
        }, ckpt_path / f'{step:d}.pth')

    hvp_solver.close()


def main(cfg):
    start_root = Path(cfg['input']['checkpoint_dir'])
    beta = torch.Tensor([1, 0])
    for start_path in sorted(start_root.glob('[0-9]*.pth'), key=lambda x: int(x.name.split('.')[0])):
        train(start_path, beta, cfg)


if __name__ == '__main__':
    args = parse_args('configs/cpmtl.yaml')
    cfg = load_config(args.config, args.overrides)
    main(cfg)
