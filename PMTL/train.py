import os
import sys
import pickle

import numpy as np
import torch
import torch.utils.data
import yaml
from torch.autograd import Variable

from models.lenet import LeNet
from models.resnet import ResNet18
from models.trainer import MTLTrainer
from solvers.min_norm_solvers import MinNormSolver
from utils.preference import circle_points


def get_d_paretomtl_init(grads, value, weights, i):
    """Compute gradient direction for the ParetoMTL initialization phase.

    Returns (flag, weight): flag=True means a feasible solution is already
    reached and no gradient step is needed for this batch.
    """
    flag = False
    nobj = value.shape

    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    if torch.sum(idx) <= 0:
        flag = True
        return flag, torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).cuda().float()
    else:
        vec = torch.matmul(w[idx], grads)
        sol, _ = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

    weight0 = torch.sum(torch.stack([sol[j] * w[idx][j, 0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 = torch.sum(torch.stack([sol[j] * w[idx][j, 1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0, weight1])

    return flag, weight


def get_d_paretomtl(grads, value, weights, i):
    """Compute gradient direction for the main ParetoMTL training phase."""
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight

    gx = torch.matmul(w, value / torch.norm(value))
    idx = gx > 0

    if torch.sum(idx) <= 0:
        sol, _ = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).cuda().float()

    vec = torch.cat((grads, torch.matmul(w[idx], grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])

    weight0 = sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2, 0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 = sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2, 1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0, weight1])

    return weight


def train(dataset, model, niter, npref, init_weight, pref_idx, data_dir, save_dir):
    n_tasks = 2
    ref_vec = torch.tensor(circle_points([1], [npref])[0]).cuda().float()

    data_files = {
        'mnist': 'multi_mnist.pickle',
        'fashion': 'multi_fashion.pickle',
        'fashion_and_mnist': 'multi_fashion_and_mnist.pickle',
    }
    with open(os.path.join(data_dir, data_files[dataset]), 'rb') as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trainX, trainLabel),
        batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(testX, testLabel),
        batch_size=256, shuffle=False)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    if model == 'lenet':
        net = MTLTrainer(LeNet(n_tasks), init_weight)
    elif model == 'resnet18':
        net = MTLTrainer(ResNet18(n_tasks), init_weight)

    if torch.cuda.is_available():
        net.cuda()

    if model == 'lenet':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 30, 45, 60, 75, 90], gamma=0.5)
    elif model == 'resnet18':
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1)

    weights = []
    task_train_losses = []
    task_test_losses = []
    train_accs = []
    test_accs = []

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'result.txt')

    ref_vec_str = ', '.join(map(str, ref_vec[pref_idx].cpu().numpy()))
    print('Preference Vector ({}/{}): {}'.format(pref_idx + 1, npref, ref_vec_str))
    with open(log_path, 'a') as f:
        f.write('Preference Vector ({}/{}): {}\n'.format(pref_idx + 1, npref, ref_vec_str))

    # Initialization phase: run at most 2 epochs to find a feasible starting point.
    # The for/else construct breaks out of both loops once feasibility is found.
    for t in range(2):
        net.train()
        for batch in train_loader:
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []

            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = net(X, ts)
                losses_vec.append(task_loss[i].data)
                task_loss[i].backward()
                grads[i] = []
                for param in net.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads, losses_vec, ref_vec, pref_idx)

            if flag:
                print("feasible solution is obtained.")
                break

            # Re-run forward pass: the graph from the gradient computation above
            # is already freed after .backward(), so we need a fresh forward pass.
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = net(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            loss_total.backward()
            optimizer.step()
        else:
            continue
        break

    # Main training phase
    for t in range(niter):
        scheduler.step()
        net.train()
        for batch in train_loader:
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            grads = {}
            losses_vec = []

            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = net(X, ts)
                losses_vec.append(task_loss[i].data)
                task_loss[i].backward()
                grads[i] = []
                for param in net.parameters():
                    if param.grad is not None:
                        grads[i].append(Variable(param.grad.data.clone().flatten(), requires_grad=False))

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)

            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads, losses_vec, ref_vec, pref_idx)

            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff

            # Re-run forward pass for the actual gradient update (same reason as init phase).
            optimizer.zero_grad()
            for i in range(len(task_loss)):
                task_loss = net(X, ts)
                if i == 0:
                    loss_total = weight_vec[i] * task_loss[i]
                else:
                    loss_total = loss_total + weight_vec[i] * task_loss[i]
            loss_total.backward()
            optimizer.step()

        if t == 0 or (t + 1) % 2 == 0:
            net.eval()
            with torch.no_grad():
                total_train_loss = []
                correct1_train = 0
                correct2_train = 0

                for batch in train_loader:
                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_train_loss = net(X, ts)
                    total_train_loss.append(valid_train_loss)
                    output1 = net.model(X).max(2, keepdim=True)[1][:, 0]
                    output2 = net.model(X).max(2, keepdim=True)[1][:, 1]
                    correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                    correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

                train_acc = np.stack([
                    1.0 * correct1_train / len(train_loader.dataset),
                    1.0 * correct2_train / len(train_loader.dataset)
                ])

                total_train_loss = torch.stack(total_train_loss)
                average_train_loss = torch.mean(total_train_loss, dim=0)

                total_test_loss = []
                correct1_test = 0
                correct2_test = 0

                for batch in test_loader:
                    X = batch[0]
                    ts = batch[1]
                    if torch.cuda.is_available():
                        X = X.cuda()
                        ts = ts.cuda()

                    valid_test_loss = net(X, ts)
                    total_test_loss.append(valid_test_loss)
                    output1 = net.model(X).max(2, keepdim=True)[1][:, 0]
                    output2 = net.model(X).max(2, keepdim=True)[1][:, 1]
                    correct1_test += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                    correct2_test += output2.eq(ts[:, 1].view_as(output2)).sum().item()

                test_acc = np.stack([
                    1.0 * correct1_test / len(test_loader.dataset),
                    1.0 * correct2_test / len(test_loader.dataset)
                ])
                total_test_loss = torch.stack(total_test_loss)
                average_test_loss = torch.mean(total_test_loss, dim=0)

            if torch.cuda.is_available():
                task_train_losses.append(average_train_loss.data.cpu().numpy())
                task_test_losses.append(average_test_loss.data.cpu().numpy())
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                weights.append(weight_vec.cpu().numpy())

                log_str = '{}/{}: weights={}, train_loss={}, train_acc={}, test_loss={}, test_acc={}'.format(
                    t + 1, niter, weights[-1], task_train_losses[-1], train_accs[-1],
                    task_test_losses[-1], test_accs[-1])
                print(log_str)
                with open(log_path, 'a') as f:
                    f.write(log_str + '\n')

    save_path = os.path.join(
        save_dir,
        '{}_{}_niter{}_npref{}_pref{}.pkl'.format(dataset, model, niter, npref, pref_idx)
    )
    torch.save(net.model.state_dict(), save_path)


def run(dataset, model, niter, npref, data_dir='data', save_dir='saved_model'):
    """Train ParetoMTL for all preference vectors."""
    init_weight = np.array([0.5, 0.5])
    for pref_idx in range(npref):
        train(dataset, model, niter, npref, init_weight, pref_idx, data_dir, save_dir)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/train.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run(**cfg)
