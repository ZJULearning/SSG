# SSG with ADA-NNS 

This repository is NSSG with greey search method (baseline) and ADA-NNS.

Please refer to original [readme](https://github.com/SNU-ARC/SSG/blob/master/README.md).

## Building Instruction

### Prerequisites
+ OpenMP
+ CMake
+ Boost

### Compile on Ubuntu:


1. Install Dependencies:

```shell
$ sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev
```

2. Compile NSSG:

```shell
$ git clone https://github.com/SNU-ARC/SSG
$ cd SSG/
$ git checkout graphANNS
$ cd build
$ ./build.sh
```

## Usage

We provide the script which can build and search for in memory-resident indices. The scripts locate under directory `tests/.` For the description of original main binaries, please refer to original [readme](https://github.com/SNU-ARC/SSG/blob/master/README.md).

### Building SSG Index

To use NSSG for ANNS, an NSSG index must be built first. Here are the instructions for building NSSG.

#### Step 1. Build kNN graph

Firstly, we need to prepare a kNN graph.

NSSG suggests to use [efanna\_graph](https://github.com/ZJULearning/efanna\_graph) to build this kNN graph. We provide the script to build kNN graphs for various datasets. Please refer to [efanna\_graph](https://github.com/SNU-ARC/efanna\_graph) and checkout `graphANNS` branch.

You can also use any alternatives, such as [Faiss](https://github.com/facebookresearch/faiss).

#### Step 2. Build NSSG index and search via NSSG index

Secondly, we will convert the kNN graph to our NSSG index and perform search.

To use the greedy search, use the `tests/evaluate_baseline.sh` script:
```shell
$ cd tests/
$ ./evaluate_baseline.sh [dataset]
```
The argument is as follows:

(i) dataset: Name of the dataset. The script supports various real datasets (e.g., SIFT1M, GIST1M, CRAWL, DEEP1M, DEEP100M_16T).

To change parameter for search (e.g., K, L, number of threads), open `evaluate_baseline.sh` and modify the parameter `K, L_SIZE, THREAD`.

To use the ADA-NNS, use the `tests/evaluate_ADA-NNS.sh` script:
```shell
$ cd tests/
$ ./evaluate_ADA-NNS.sh [dataset]
```
The arguments is same as above in `evaluate_baseline.sh`.

To change parameter for search (e.g., K, L, number of threads), open `evaluate_ADA-NNS.sh` and modify the parameter `K, L_SIZE, THREAD`.
