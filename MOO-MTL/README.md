# Multi-Task Learning as Multi-Objective Optimization

This code repository includes the source code for the [Paper](https://arxiv.org/abs/1810.04650):

```
Multi-Task Learning as Multi-Objective Optimization
Ozan Sener, Vladlen Koltun
Neural Information Processing Systems (NeurIPS) 2018 
```

The experimentation framework is based on PyTorch; however, the proposed algorithm (MGDA_UB) is implemented largely Numpy with no other requirement. So, it should be trivial to extend to other deep learning frameworks. PyTorch version is implemented in `min_norm_solvers.py`, generic version using only Numpy is implemented in file `min_norm_solvers_numpy.py`.

This repo includes more than the implementation of the paper. It imlpements both Frank-Wolfe and projected gradient descent method. It also has smart initialization and gradient normalization tricks which are described with inline comments.

The source code and dataset (MultiMNIST) are released under the MIT License. See the License file for details.


# Requirements and References
The code uses the following Python packages and they are required: ``pytorch, torchvision, numpy, scipy, Pillow, tqdm, PyYAML, six``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

We adapt and use some code snippets from:
* [CSAILVision Semanti Segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch)
* [PyTorch-SemSeg](https://github.com/meetshah1995/pytorch-semseg/)



# Usage
The code base uses `configs/config.yaml` for global configurations, dataset paths, optimizer settings, and training parameters.

To train a model, use the command: 
```bash
python train.py --config configs/config.yaml
```
