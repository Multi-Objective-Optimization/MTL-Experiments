# MultiMNIST

We use the data provided in [Pareto Multi-Task Learning](https://papers.nips.cc/paper/9374-pareto-multi-task-learning).
The data is available [here](https://drive.google.com/drive/folders/1VnmCmBAVh8f_BKJg1KYx-E137gBLXbGG).

Please create a `data` folder and download the `.pickle` files. 

## Run experiments

We support two model variants: `resnet` and `lenet` controlled by the `--model` flag. To run the LeNet experiment, use e.g.,

```bash
python train_multimnist.py --datapath data/multi_fashion_and_mnist.pickle --model lenet
```

For the ResNet experiment with 5M trainable parameters, use e.g.,

```bash
python train_multimnist.py --datapath data/multi_fashion_and_mnist.pickle --model resnet --resnet-size 5M
```

We also support 1M, 2M and 11M trainable parameters.


# SARCOS

## Dataset

The SARCOS data is available [here](http://gaussianprocess.org/gpml/data). Please create a `data` folder and download the `.mat` files.

## Run Experiment

To train PHN-EPO run:

```bash
python train_sarcos.py --datapath data --solevr epo
```