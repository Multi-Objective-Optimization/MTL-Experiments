import os
import yaml
import torch
import argparse
from timeit import default_timer as timer

from torch.autograd import Variable

from tqdm import tqdm

import losses
import datasets
import metrics
import model_selector
from solvers.min_norm_solver import MinNormSolver, gradient_normalizers


def train(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg.update(cfg.pop('datasets')[cfg['dataset']])

    exp_id = cfg['dataset']

    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)

    train_loader, _, val_loader, val_dst = datasets.get_dataset(cfg)
    loss_fn = losses.get_loss(cfg)
    metric = metrics.get_metrics(cfg)

    model, device = model_selector.get_model(cfg)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    optimizer_type = cfg['optimizer']['type']
    lr = cfg['optimizer']['lr']
    if 'RMSprop' in optimizer_type:
        optimizer = torch.optim.RMSprop(model_params, lr=lr)
    elif 'Adam' in optimizer_type:
        optimizer = torch.optim.Adam(model_params, lr=lr)
    elif 'SGD' in optimizer_type:
        optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9)

    tasks = cfg['tasks']
    all_tasks = cfg['all_tasks']
    lr_decay_factor = cfg['training']['lr_decay']['factor']
    lr_decay_step = cfg['training']['lr_decay']['step']
    num_epochs = cfg['training']['epochs']
    solver_max_iter = cfg['solver']['max_iter']
    solver_stop_crit = cfg['solver']['stop_criterion']

    print('Starting training with config \n \t{} \n'.format(config))

    if 'mgda' in cfg['algorithm']['type']:
        approximate_norm_solution = cfg['algorithm']['use_approximation']
        if approximate_norm_solution:
            print('Using approximate min-norm solver')
        else:
            print('Using full solver')

    n_iter = 0
    for epoch in tqdm(range(num_epochs)):
        start = timer()
        print('Epoch {} Started'.format(epoch))
        if (epoch+1) % lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
            print('Decayed learning rate at iter {}'.format(n_iter))

        for m in model:
            model[m].train()

        for batch in train_loader:
            n_iter += 1
            images = batch[0].to(device)

            labels = {}
            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels[t] = batch[i+1].to(device)

            loss_data = {}
            grads = {}
            scale = {}
            mask = None
            masks = {}

            if 'mgda' in cfg['algorithm']['type']:
                if approximate_norm_solution:
                    optimizer.zero_grad()
                    with torch.no_grad():
                        rep, mask = model['rep'](images, mask)
                    if isinstance(rep, list):
                        rep = rep[0]
                        rep_variable = [rep.detach().clone().requires_grad_(True)]
                        list_rep = True
                    else:
                        rep_variable = rep.detach().clone().requires_grad_(True)
                        list_rep = False

                    for t in tasks:
                        optimizer.zero_grad()
                        out_t, masks[t] = model[t](rep_variable, None)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []
                        if list_rep:
                            grads[t].append(rep_variable[0].grad.detach().clone())
                            rep_variable[0].grad.zero_()
                        else:
                            grads[t].append(rep_variable.grad.detach().clone())
                            rep_variable.grad.zero_()
                else:
                    for t in tasks:
                        optimizer.zero_grad()
                        rep, mask = model['rep'](images, mask)
                        out_t, masks[t] = model[t](rep, None)
                        loss = loss_fn[t](out_t, labels[t])
                        loss_data[t] = loss.item()
                        loss.backward()
                        grads[t] = []
                        for param in model['rep'].parameters():
                            if param.grad is not None:
                                grads[t].append(param.grad.detach().clone())

                gn = gradient_normalizers(grads, loss_data, cfg['algorithm']['normalization'])
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                sol, _ = MinNormSolver.find_min_norm_element(
                    [grads[t] for t in tasks],
                    max_iter=solver_max_iter,
                    stop_crit=solver_stop_crit
                )
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
            else:
                for t in tasks:
                    masks[t] = None
                    scale[t] = float(cfg['scales'][t])

            optimizer.zero_grad()
            rep, _ = model['rep'](images, mask)
            for i, t in enumerate(tasks):
                out_t, _ = model[t](rep, masks[t])
                loss_t = loss_fn[t](out_t, labels[t])
                loss_data[t] = loss_t.item()
                if i > 0:
                    loss = loss + scale[t]*loss_t
                else:
                    loss = scale[t]*loss_t
            loss.backward()
            optimizer.step()

        for m in model:
            model[m].eval()

        tot_loss = {}
        tot_loss['all'] = 0.0
        for t in tasks:
            tot_loss[t] = 0.0

        num_val_batches = 0
        with torch.no_grad():
            for batch_val in val_loader:
                val_images = batch_val[0].to(device)
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i+1].to(device)

                val_rep, _ = model['rep'](val_images, None)
                for t in tasks:
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    loss_value = loss_t.item()
                    tot_loss['all'] += loss_value
                    tot_loss[t] += loss_value
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches += 1

        print('Epoch {} | val_loss: {:.4f}'.format(epoch, tot_loss['all'] / num_val_batches))
        for t in tasks:
            metric_results = metric[t].get_result()
            print('  task {} | loss: {:.4f} | {}'.format(
                t, tot_loss[t] / num_val_batches,
                ' | '.join('{}: {:.4f}'.format(k, v) for k, v in metric_results.items())
            ))
            metric[t].reset()

        if epoch % 3 == 0:
            state = {'epoch': epoch+1,
                    'model_rep': model['rep'].state_dict(),
                    'optimizer_state': optimizer.state_dict()}
            for t in tasks:
                state['model_{}'.format(t)] = model[t].state_dict()

            torch.save(state, os.path.join(
                cfg['paths']['checkpoint_dir'],
                '{}_{}_model.pkl'.format(exp_id, epoch+1)
            ))

        end = timer()
        print('Epoch ended in {}s'.format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)
