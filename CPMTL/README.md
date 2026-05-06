# Efficient Continuous Pareto Exploration in Multi-Task Learning

[Pingchuan Ma](https://pingchuan.ma/)\*,
[Tao Du](https://people.csail.mit.edu/taodu/)\*,
and
[Wojciech Matusik](http://people.csail.mit.edu/wojciech/)

**ICML 2020**
[[Project Page]](http://cpmtl.csail.mit.edu/)
[[Paper]](https://arxiv.org/abs/2006.16434)
[[Video]](https://icml.cc/virtual/2020/poster/5856)
[[Slides]](http://cpmtl.csail.mit.edu/data/slides.pdf)

```text
@inproceedings{ma2020efficient,
    title={Efficient Continuous Pareto Exploration in Multi-Task Learning},
    author={Ma, Pingchuan and Du, Tao and Matusik, Wojciech},
    booktitle={International Conference on Machine Learning},
    pages={6522--6531},
    year={2020},
    organization={PMLR}
}
```

## Quick Start

Online demos for MultiMNIST and UCI-Census are available in Google Colab! Try them now!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-gfx/ContinuousParetoMTL/blob/colab)

## Example for MultiMNIST

We provide an example for MultiMNIST dataset. First, we run weighted sum method for initial Pareto solutions:

```sh
python weighted_sum.py
```

The output should be like:

```text
0: loss [2.313036/2.304537] top@1 [7.65%/10.65%]
0: 1/30: loss [1.463346/0.909529] top@1 [51.52%/69.72%]
0: 2/30: loss [0.889257/0.638646] top@1 [71.29%/78.55%]
0: 3/30: loss [0.703745/0.534612] top@1 [77.77%/81.86%]
0: 4/30: loss [0.622291/0.491764] top@1 [80.13%/83.02%]
```

Based on these starting solutions, we can run our continuous Pareto exploration by:

```sh
python cpmtl.py
```

The output should be like:

```text
0: 1/10: loss [0.397692/0.350267] top@1 [86.57%/88.11%]
    86.37% 86.57% Δ=0.20% absΔ=0.20%
    88.10% 88.11% Δ=0.01% absΔ=0.01%

0: 2/10: loss [0.392314/0.351280] top@1 [86.85%/88.07%]
    86.37% 86.57% 86.85% Δ=0.28% absΔ=0.48%
    88.10% 88.11% 88.07% Δ=-0.04% absΔ=-0.03%

0: 3/10: loss [0.387585/0.352643] top@1 [86.92%/88.03%]
    86.37% 86.57% 86.85% 86.92% Δ=0.07% absΔ=0.55%
    88.10% 88.11% 88.07% 88.03% Δ=-0.04% absΔ=-0.07%
```

Now you can play it on your own dataset and network architecture!
