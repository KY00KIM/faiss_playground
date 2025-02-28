import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from utils import get_index_list, get_index_prefix, ntfy


def sample_train_vecs(dataset: NDArray, sample_ratio: float) -> NDArray:
    assert sample_ratio > 1 or sample_ratio < 0, "sample ratio must be between 0~1"
    num_samples = int(sample_ratio * dataset.shape[0])
    indicies = np.random.choice(dataset.shape[0], num_samples, replace=False)
    return dataset[indicies]


def random_sample_array(array: NDArray, sample_ratio=0.8, random_seed=None) -> NDArray:
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be between 0 and 1.")
    if random_seed is not None:
        np.random.seed(random_seed)
    rand_indicies = np.random.permutation(len(array))[: int(len(array) * sample_ratio)]
    return array[rand_indicies]


def is_valid_gpu_index(index_name: str) -> bool:
    name_uc = index_name.upper()

    # IVF_HNSW not implemented for gpu
    if "IVF" in name_uc and "HNSW" in name_uc:
        return False

    # PQ not implemented for gpu except for IVFPQ
    if "PQ" in name_uc and "IVF" not in name_uc:
        return False

    return True


def train_idx(
    index_name: str,
    dim: int,
    train_vec: NDArray,
    save: bool,
    save_path: Union[None, str],
    n_gpu: int = 0,
) -> Tuple[faiss.Index, Dict]:
    index = faiss.index_factory(dim, index_name)

    if n_gpu > 0 and is_valid_gpu_index(index_name):
        assert (
            faiss.StandardGpuResources
        ), "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
        params = faiss.GpuParameterSpace()
        if n_gpu > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            # if "IVF" in index_name:
            #     co.common_ivf_quantizer = True
            index = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=n_gpu)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        params = faiss.ParameterSpace()

    params.initialize(index)
    params.verbose = True

    hw_str = f"gpu({n_gpu})" if n_gpu > 0 else f"cpu({faiss.omp_get_max_threads()})"
    train_info = {
        "hw": hw_str,
        "dim": train_vec.shape[1],
        "num_sample": train_vec.shape[0],
    }
    print(f"Training {index_name} w/ {train_vec.shape[0]} on {hw_str}")
    t0 = time.perf_counter()
    index.train(train_vec)
    t1 = time.perf_counter()
    train_info["time"] = t1 - t0
    print(f"Training {index_name} took {t1-t0} sec")

    if n_gpu > 0:
        index = faiss.index_gpu_to_cpu(index)

    if save:
        faiss.write_index(index, save_path)
    return (index, train_info)


def train_all(
    index_list: List[str],
    embs: NDArray,
    train_ratio: float,
    dataset_prefix: str,
    output_dir: str,
    train_info_path: str,
    n_gpu: int,
    n_thread: int,
):
    assert (
        n_gpu <= faiss.get_num_gpus()
    ), f"NUM_GPU={n_gpu} must be less or equal to {faiss.get_num_gpus()} gpus"
    assert (
        n_thread > 0 and n_thread <= faiss.omp_get_max_threads()
    ), f"NUM_THREAD should be between 0 to {faiss.omp_get_max_threads()}"
    assert train_ratio > 0 and train_ratio <= 1, "TRAIN_RATIO must be between 0 and 1"
    embs = embs[: int(embs.shape[0] * train_ratio)]
    dim = embs.shape[1]
    train_data = []
    n_max_thread = faiss.omp_get_max_threads()
    if n_gpu < 1:
        faiss.omp_set_num_threads(n_thread)
    for i, index in tqdm(enumerate(index_list), desc="index train"):
        try:
            _, info = train_idx(
                index_name=index,
                dim=dim,
                train_vec=embs,
                save=True,
                save_path=os.path.join(
                    output_dir, f"{dataset_prefix}_{get_index_prefix(index)}.faiss"
                ),
                n_gpu=n_gpu,
            )
            train_data.append({"index": index, **info})
        except Exception as e:
            if n_gpu > 0:
                print("Fallback to CPU")
                _, info = train_idx(
                    index_name=index,
                    dim=dim,
                    train_vec=embs,
                    save=True,
                    save_path=os.path.join(
                        output_dir, f"{dataset_prefix}_{get_index_prefix(index)}.faiss"
                    ),
                    n_gpu=0,
                )
                train_data.append({"index": index, **info})
            else:
                train_info = {"index": index}
                train_info["error"] = str(e)
                train_data.append(train_info)
        ntfy(
            f"{index} training done {i+1}/{len(index_list)}: \ndetail:{train_data[-1]}"
        )
    if n_gpu < 1:
        faiss.omp_set_num_threads(n_max_thread)
    print("Train Summary=============")
    for row in train_data:
        print(row)
    print("==========================")
    err_idx = []
    for i in train_data:
        if "error" in i.keys():
            err_idx.append(i["index"])
    print(f"Error {len(err_idx)}/{len(index_list)} : {err_idx}")
    with open(train_info_path, "w") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")


def add_idx(index, base_vec: NDArray):
    index.add(base_vec)
    return index


def add_idx_all_dir(index_base_dir: str, index_output_dir: str, embs: NDArray):
    index_paths = [f for f in os.listdir(index_base_dir) if f.endswith(".faiss")]

    for i, index_filename in tqdm(enumerate(index_paths), desc="Populate Index"):
        output_path = os.path.join(index_output_dir, index_filename)
        if os.path.exists(output_path):
            continue
        t0 = time.perf_counter()
        index = faiss.read_index(os.path.join(index_base_dir, index_filename))
        t1 = time.perf_counter()
        populated_idx = add_idx(index, embs)
        faiss.write_index(populated_idx, output_path)
        populated_idx = faiss.read_index(output_path)
        if populated_idx.ntotal != embs.shape[0]:
            print(
                "Error: ",
                output_path,
                "contains",
                populated_idx.ntotal,
                "/",
                embs.shape[0],
            )
            ntfy(
                f"{index_filename} Add failed {i+1}/{len(index_paths)}\ntime:{t1-t0}sec\npath:{index_base_dir}"
            )
        else:
            ntfy(
                f"{index_filename} Add done {i+1}/{len(index_paths)}\ntime:{t1-t0}sec\npath:{index_base_dir}"
            )


def main(args):
    assert args.embedding_path.exists(), "Embedding path does not exists"
    embs = np.load(args.embedding_path, mmap_mode="r")
    emb_size = embs.shape[0]
    assert args.embedding_num == -1 or (
        args.embedding_num > 0 and args.embedding_num <= emb_size
    ), f"Embedding num should be -1 or between 0 and {emb_size}"
    embs = embs[: args.embedding_num]

    if args.process == "train" or args.process == "all":
        assert (
            0 < args.train_ratio and args.train_ratio <= 1
        ), f"Error: train_ratio should be between 0 and 1, got {args.train_ratio}"
        assert args.index, "Error: At least one index must be provided for training."
        assert args.num_gpu >= 0, "Error: num_gpu should be non-negative."
        assert args.num_thread > 0, "Error: num_thread must be greater than 0."
        os.makedirs(args.index_base_dir, exist_ok=True)
        train_all(
            index_list=args.index,
            embs=embs,
            train_ratio=args.train_ratio,
            dataset_prefix=args.output_prefix,
            output_dir=args.index_base_dir,
            train_info_path=args.train_jsonl_path,
            n_gpu=args.num_gpu,
            n_thread=args.num_thread,
        )

    if args.process == "add" or args.process == "all":
        os.makedirs(args.index_output_dir, exist_ok=True)
        add_idx_all_dir(
            index_base_dir=args.index_base_dir,
            index_output_dir=args.index_output_dir,
            embs=embs,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument(
        "--process", type=str, choices=["train", "add", "all"], required=True
    )
    parser.add_argument(
        "--index-base-dir",
        type=Path,
        required=True,
        help="Directory for empty trained indexes",
    )
    parser.add_argument(
        "--embedding-path", type=Path, required=True, help="Embedding path to use"
    )
    parser.add_argument(
        "--embedding-num",
        type=int,
        default=-1,
        required=True,
        help="Number of embedding to use -1 for all",
    )
    # population args
    parser.add_argument(
        "--index-output-dir", type=Path, help="Directory for populated indexes"
    )
    # train args
    parser.add_argument(
        "--index",
        "-i",
        choices=get_index_list(),
        nargs="+",
        help="List of indexes to process",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of embedding to use for training index",
    )
    parser.add_argument(
        "--output-prefix", type=str, help="Prefix for trained index files"
    )
    parser.add_argument(
        "--train-jsonl-path", type=Path, help="Path for training info jsonl"
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=0,
        help="Number of GPU threads for training, 0 for CPU",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=faiss.omp_get_max_threads(),
        help="Number of threads to use for training",
    )
    args = parser.parse_args()

    print("PROVIDED ARGS===============")
    print(args)
    print("============================")

    main(args)
