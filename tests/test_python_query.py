import os
import time
import argparse
import numpy as np

import pyssg


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str,
                        default="./datasets/sift/sift_base.fvecs",
                        help="fvecs file for base vectors")
    parser.add_argument("--query", type=str,
                        default="./datasets/sift/sift_query.fvecs",
                        help="fvecs file for query vectors")
    parser.add_argument("--groundtruth", type=str,
                        default="./datasets/sift/sift_groundtruth.ivecs",
                        help="ivecs file for groundtruth")
    parser.add_argument("--graph", type=str,
                        default="./graphs/sift.ssg",
                        help="path SSG graph file")
    parser.add_argument("--k", type=int, default=100,
                        help="how many neighbors to query")
    parser.add_argument("--l", type=int, default=100,
                        help="search param L")
    parser.add_argument("--seed", type=int, default=161803398,
                        help="random seed")
    return parser.parse_args()


def load_vecs(filename):
    _, ext = os.path.splitext(filename)
    if ext == ".fvecs":
        dtype = np.float32
    elif ext == ".ivecs":
        dtype = np.int32
    else:
        raise TypeError("Unknown file type: {}".format(ext))

    data = np.fromfile(filename, dtype=dtype)
    dim = data[0].view(np.int32)
    data = data.reshape(-1, dim + 1).astype(dtype)
    return np.ascontiguousarray(data[:, 1:])


if __name__ == "__main__":
    args = setup_args()

    base = load_vecs(args.base)
    nbases, dim = base.shape

    query = load_vecs(args.query)
    nq, _ = query.shape

    pyssg.set_seed(args.seed)
    index = pyssg.IndexSSG(dim, nbases)
    index.load(args.graph, base)

    start = time.time()
    results = [index.search(q, args.k, args.l) for q in query]
    elapsed = time.time() - start
    qps = nq / elapsed

    results = np.asarray(results)
    gt = load_vecs(args.groundtruth)[:nq]
    assert gt.shape == results.shape

    cnt = 0
    for ret, gt in zip(results, gt):
        cnt += len(np.intersect1d(ret, gt))
    acc = cnt / nq / args.k * 100
    print(f"{nq} queries in {elapsed:.4f}s, {qps:.4f}QPS, accuracy {acc:.4f}")


