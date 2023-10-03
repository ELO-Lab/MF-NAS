# MF-NAS: Multi-Fidelity Neural Architecture Search
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
- **MF-NAS** (**Ours**)

Our experiments are conducted on **NAS-Bench-101**, **NAS-Bench-201**, and **NAS-Bench-ASR** search spaces. 
The configurations of algorithms and problems are set in [`configs/algo.yaml`](configs/algo.yaml) and [`configs/problem.yaml`](configs/problem.yaml), respectively.

To reproduce all results in our paper, follow the below table to set the correct hyperparameters:
| Algorithm                   | Hyperparameters            |            
|:--------------------------|:----------------------|
| RS / FLS / BLS          | <ul><li>**using_zc_metric**: _False_</li><li>**metric**: _val_acc_ (for **NB-101**, **NB-201**) / _val_per_ (for **NB-ASR**)</li><li>**iepoch**: $12$</li> |
| SH         | <ul><li>**using_zc_metric**: _False_</li><li>**metric**: _val_acc_ (for **NB-101**, **NB-201**) / _val_per_ (for **NB-ASR**)</li><li>**list_iepoch**: <ul><li>$[4, 12, 36, 108]$ (for **NB-101**)</li><li>$[12, 25, 50, 100, 200]$ (for **NB-201**)</li><li>$[10, 20, 30, 40]$ (for **NB-ASR**)</li></ul><li>**n_candidate**: $16$ (for **NB-101, NB-ASR**) / $32$ (for **NB-201**)</li>|
| REA / REA+W          | <ul><li>**using_zc_metric**: _False_</li><li>**metric**: _val_acc_ (for **NB-101**, **NB-201**) / _val_per_ (for **NB-ASR**)</li><li>**iepoch**: $12$</li><li>**pop_size**: $10$</li><li>**tournament_size**: $10$</li><li>**prob_mutation**: $1.0$</li><li>**warm_up**: _True_ (for **REA+W**)</li><li>**n_sample_warmup**: $2000$ (for **REA+W**)</li><li>**metric_warmup**: _synflow_ (for **REA+W**)</li>|
| MF-NAS         | <ul><li>**metric_stage1**: _params_</li><li>**optimizer_stage1**: **FLS** / **BLS** (default: **FLS**)</li><li>**using_zc_metric_stage1**: _True_</li><li>**max_eval_stage1**: $2000$</li><li>**using_zc_metric_stage2**: _False_</li><li>**metric_stage2**: _val_acc_ (for **NB-101**, **NB-201**) / _val_per_ (for **NB-ASR**)</li><li>**list_iepoch**: <ul><li>$[4, 12, 36, 108]$ (for **NB-101**)</li><li>$[12, 25, 50, 100, 200]$ (for **NB-201**)</li><li>$[10, 20, 30, 40]$ (for **NB-ASR**)</li></ul><li>**k**: $16$ (for **NB-101, NB-ASR**) / $32$ (for **NB-201**)</li>|

and run the below script:
```shell
$ python main.py --ss <search-space> --optimizer <search-strategy> --n_run <number-of-runs> 500
```
| Args                   | Choices            |            
|:--------------------------|:----------------------|
| _ss_          | **nb101**, **nb201**, **nbasr** |
| _optimizer_         | **RS**, **FLS**, **BLS**, **SH**, **REA**, **REA+W**, **MF-NAS**|

Note that you can use our implemented algorithms and search with other metrics. However, the `using_zc_metric` and `metric` hyperparameters need to be set so that they do not conflict with each other.

For example, if you use the **synflow** or **jacov** metrics, you need to set `using_zc_metric` to `True`.
If you use **val_acc** as the search objective, you must set `using_zc_metric` to `False`.

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
|MF-NAS (_synflow_)          | $93.82 \pm 0.56$<br/>$12,960$ seconds<br/>($368$ epochs)|$94.36 \pm 0.05$<br/>$20,000$ seconds<br/>($668$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs) | $21.77 \pm 0.00$<br/>$300$ epochs |
|MF-NAS (_params_)          | $93.89 \pm 0.25$<br/>$14,088$ seconds<br/>($368$ epochs)  |$94.36 \pm 0.00$<br/>$20,000$ seconds<br/>($617$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs) | $21.81 \pm 0.26$<br/>$300$ epochs |
|MF-NAS (_FLOPS_)          | $93.88 \pm 0.25$<br/>$14,055$ seconds<br/>($368$ epochs)  |$94.36 \pm 0.00$<br/>$20,000$ seconds<br/>($617$ epochs)| $73.51 \pm 0.00$<br/>$40,000$ seconds<br/>($1,192$ epochs) | $46.34 \pm 0.00$<br/>$120,000$ seconds<br/>($1,192$ epochs)| $21.78 \pm 0.36$<br/>$300$ epochs |
|Optimal (_benchmark_)          | $94.31$  | $94.37$ | $73.51$  | $47.31$ | $21.40$ |

## Acknowledgement
We want to give our thanks to the authors of [NAS-Bench-101](https://arxiv.org/abs/1902.09635), [NAS-Bench-201](https://arxiv.org/abs/2001.00326), and [NAS-Bench-ASR](https://openreview.net/forum?id=CU0APx9LMaL) for their search spaces; to the authors of [Zero-cost Lightweight NAS](https://openreview.net/pdf?id=0cmMMy8J5q) and [NAS-Bench-Zero-Suite](https://openreview.net/pdf?id=yWhuIjIjH8k) for their zero-cost metric databases.