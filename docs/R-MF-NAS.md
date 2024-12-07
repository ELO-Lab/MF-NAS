# R-MF-NAS: A robust variant of Multi-Fidelity Neural Architecture Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong

## Setup
- Clone this repo
- Install necessary packages and databases.
```
$ cd MF-NAS
$ bash install.sh
```
## Reproducing the results
This repo have already implemented following NAS algorithms:
- **Random Search (RS)**
- **Local Search**: **First-improvement Local Search (FLS)**; **Best-improvement Local Search (BLS)**
- [**Successive Halving (SH)**](http://proceedings.mlr.press/v51/jamieson16.pdf)
- [**Regularized Evolution (REA)**](https://dl.acm.org/doi/abs/10.1609/aaai.v33i01.33014780)
- [**REA + Warmup (REA+W)**](https://openreview.net/pdf?id=0cmMMy8J5q) 
- [**MF-NAS**](https://dl.acm.org/doi/10.1145/3638529.3654027)
- **R-MF-NAS** (**Ours**)

Our experiments are conducted on **NAS-Bench-101**, **NAS-Bench-201**, and **NAS-Bench-ASR** search spaces.

The configurations of algorithms are set in [`configs/algo_101.yaml`](../configs/algo_101.yaml), [`configs/algo_201.yaml`](../configs/algo_201.yaml), and [`configs/algo_asr.yaml`](../configs/algo_asr.yaml).
The configurations of problems are set in [`configs/problem.yaml`](../configs/problem.yaml).

To reproduce all results (i.e., experiments with different ZC metrics and varying k values) in our paper, run the below scripts:
```shell
$ python /script/run_101_rmfnas.sh
$ python /script/run_201_rmfnas.sh
$ python /script/run_asr_rmfnas.sh
```

Note that you can search with other metrics. However, the `using_zc_metric` and `metric` hyperparameters must be set so that they do not conflict with each other.

For example, if you use the **synflow**/**jacov** metrics as the search objective, you need to set `using_zc_metric` to `True`.
If you use **val_acc**/**train_loss** as the search objective, you must set `using_zc_metric` to `False`.

Here are the performance comparisons between MF-NAS and R-MF-NAS with different ZC metrics for all experimental problems in our article:

| **Performance**         | **MF-NAS**                 | **R-MF-NAS**                     |
|--------------------------|----------------------------|-----------------------------------|
| **_NB101-CF10_** ↑      |                            |                                   |
| **Best**                | _93.89 ± 0.25_            | **93.92 ± 0.23**                 |
| **Worst**               | _84.01 ± 1.50_            | **91.86 ± 1.24**                 |
| **Mean ± SD**           | 90.18 ± 4.20              | **92.95 ± 0.84**                 |
| **Gap (Best, Worst)**   | 9.88                      | **2.06**                         |
| **_NB201-CF10_** ↑      |                            |                                   |
| **Best**                | **94.36 ± 0.00**          | _94.32 ± 0.08_                   |
| **Worst**               | _89.42 ± 1.04_            | **92.90 ± 0.68**                 |
| **Mean ± SD**           | 93.13 ± 1.26              | **93.70 ± 0.35**                 |
| **Gap (Best, Worst)**   | 4.94                      | **1.40**                         |
| **_NB201-CF100_** ↑     |                            |                                   |
| **Best**                | **73.51 ± 0.00**          | **73.51 ± 0.00**                 |
| **Worst**               | _60.72 ± 0.75_            | **69.72 ± 1.19**                 |
| **Mean ± SD**           | 69.96 ± 3.45              | **71.46 ± 1.05**                 |
| **Gap (Best, Worst)**   | 12.79                     | **3.79**                         |
| **_NB201-IN16_** ↑      |                            |                                   |
| **Best**                | **46.48 ± 0.21**          | _46.41 ± 0.14_                   |
| **Worst**               | _35.96 ± 0.31_            | **43.03 ± 2.07**                 |
| **Mean ± SD**           | 42.61 ± 3.98              | **45.06 ± 0.89**                 |
| **Gap (Best, Worst)**   | 10.52                     | **3.38**                         |
| **_NBASR-TIMIT_** ↓     |                            |                                   |
| **Best**                | **21.77 ± 0.00**          | 21.78 ± 0.08                     |
| **Worst**               | _83.14 ± 7.99_            | **23.28 ± 0.74**                 |
| **Mean ± SD**           | 29.41 ± 19.01             | **22.42 ± 0.42**                 |
| **Gap (Best, Worst)**   | 61.37                     | **1.50**                         |


## Acknowledgement
We want to give our thanks to the authors of [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), and [NAS-Bench-ASR](https://openreview.net/forum?id=CU0APx9LMaL) for their search spaces; to the authors of [Zero-cost Lightweight NAS](https://openreview.net/pdf?id=0cmMMy8J5q) and [NAS-Bench-Zero-Suite](https://openreview.net/pdf?id=yWhuIjIjH8k) for their zero-cost metric databases.
