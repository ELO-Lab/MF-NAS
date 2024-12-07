# MF-NAS: Multi-Fidelity Neural Architecture Search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong

In [**GECCO 2024**](https://dl.acm.org/doi/10.1145/3638529.3654027).

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
- **MF-NAS** (**Ours**)

Our experiments are conducted on **NAS-Bench-101**, **NAS-Bench-201**, and **NAS-Bench-ASR** search spaces.

The configurations of algorithms are set in [`configs/algo_101.yaml`](../configs/algo_101.yaml), [`configs/algo_201.yaml`](../configs/algo_201.yaml), and [`configs/algo_asr.yaml`](../configs/algo_asr.yaml).
The configurations of problems are set in [`configs/problem.yaml`](../configs/problem.yaml).

To reproduce all main results in our paper, run the below scripts:
```shell
$ python /script/run_101_mfnas.sh
$ python /script/run_201_mfnas.sh
$ python /script/run_asr_mfnas.sh
```

To reproduce the ablation studies, run the below scripts:
#### To experiment on NAS-Bench-201 with different zero-cost metrics
```shell
$ python /script/run_201_with_different_zc_metrics_mfnas.sh
```
#### To compare the impact of Random Search and Local Search on the performance of MF-NAS
```shell
$ python /script/compare_RS_and_LS_mfnas.sh
```
#### To replace `val_acc` with `train_loss` in MF-NAS
```shell
$ python /script/replace_val_acc_with_train_loss_mfnas.sh
```
Note that you can search with other metrics. However, the `using_zc_metric` and `metric` hyperparameters must be set so that they do not conflict with each other.

For example, if you use the **synflow**/**jacov** metrics as the search objective, you need to set `using_zc_metric` to `True`.
If you use **val_acc**/**train_loss** as the search objective, you must set `using_zc_metric` to `False`.

The table of performance metrics that are currently available for all networks in each search space.
| Metric                   | Type | NAS-Bench-101            |  NAS-Bench-201                         | NAS-Bench-ASR                          |            
|:--------------------------|:----------------------:|:--:|:---------------------------------------:|:---------------------------------------:|
|`training accuracy`          | training-based |:heavy_check_mark:| :heavy_check_mark: | :x: |
|`validation accuracy`        | training-based |:heavy_check_mark:| :heavy_check_mark: | :x: |
|`training loss`           | training-based |:x: | :heavy_check_mark: | :x: |
|`validation loss`         | training-based |:x: | :heavy_check_mark: | :x: |
|`validation PER`         | training-based |:x: | :x: | :heavy_check_mark:|
||||||
|[`jacov`](https://arxiv.org/abs/2006.04647v1)         | training-free |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|[`plain`](https://openreview.net/pdf?id=0cmMMy8J5q)         |  training-free |:x: | :heavy_check_mark: | :heavy_check_mark: |
|[`grasp`](https://openreview.net/forum?id=SkgsACVKPH)          | training-free |:heavy_check_mark:| :heavy_check_mark: | :x: |
|[`fisher`](https://arxiv.org/abs/1906.04113)         |  training-free |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:|
|[`epe_nas`](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_44)        |  training-free |:x: | :heavy_check_mark: | :x: |
|[`grad_norm`](https://openreview.net/pdf?id=0cmMMy8J5q)      |  training-free |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|[`snip`](https://openreview.net/forum?id=B1VZqjAcYX)        |  training-free |:heavy_check_mark:| :heavy_check_mark: | :heavy_check_mark: |
|[`synflow`](https://arxiv.org/abs/2006.05467)     |  training-free |:heavy_check_mark: |:heavy_check_mark: | :heavy_check_mark: |
|[`l2_norm`](https://openreview.net/pdf?id=0cmMMy8J5q)      |  training-free |:x: | :heavy_check_mark: | :heavy_check_mark: |
|[`zen`](https://ieeexplore.ieee.org/document/9711186)        |  training-free |:x: | :heavy_check_mark: | :x: |
|[`nwot`](https://arxiv.org/abs/2006.04647)        |  training-free |:x: | :heavy_check_mark: | :x: |
|`params`      |  training-free |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|`flops`        |  training-free |:heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

All metrics are logged as `.pickle` and `.json` files. 

Here are our best results (performance & search cost) for each search space:

| Algorithm                 | NB-101            | NB-201<br/>(cifar10) | NB-201<br/>(cifar100) | NB-201<br/>(ImageNet16-120) | NAS-Bench-ASR     |            
|:--------------------------|:-:|:-:|:-:|:-:|:-:|
|MF-NAS (_synflow_)          | $93.82 \pm 0.56$<br/>$12,960$ seconds<br/>($368$ epochs)|$94.36 \pm 0.05$<br/>$20,000$ seconds<br/>($1,192$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs) | $21.77 \pm 0.00$<br/>$300$ epochs |
|MF-NAS (_params_)          | $93.89 \pm 0.25$<br/>$14,088$ seconds<br/>($368$ epochs)  |$94.36 \pm 0.00$<br/>$20,000$ seconds<br/>($1,192$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs) | $21.81 \pm 0.26$<br/>$300$ epochs |
|MF-NAS (_FLOPS_)          | $93.88 \pm 0.25$<br/>$14,055$ seconds<br/>($368$ epochs)  |$94.36 \pm 0.00$<br/>$20,000$ seconds<br/>($1,192$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs)| $21.78 \pm 0.36$<br/>$300$ epochs |
|Optimal (_benchmark_)          | $94.31$  | $94.37$ | $73.51$  | $47.31$ | $21.40$ |

## Acknowledgement
We want to give our thanks to the authors of [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), and [NAS-Bench-ASR](https://openreview.net/forum?id=CU0APx9LMaL) for their search spaces; to the authors of [Zero-cost Lightweight NAS](https://openreview.net/pdf?id=0cmMMy8J5q) and [NAS-Bench-Zero-Suite](https://openreview.net/pdf?id=yWhuIjIjH8k) for their zero-cost metric databases.
