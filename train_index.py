import os
import time
from typing import Dict, Union, Tuple
import json
from tqdm import tqdm
import numpy as np
import faiss
from numpy.typing import NDArray
from utils import get_index_prefix, get_index_list, load_embedding


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


def train_idx(
    index_name: str,
    use_gpu: bool,
    dim: int,
    train_vec: NDArray,
    save: bool,
    save_path: Union[None, str],
) -> Tuple[faiss.Index, faiss.ParameterSpace, Dict]:
    index = faiss.index_factory(dim, index_name)
    keys_cpu = [
        "IVF65536_HNSW32",  # error
    ]

    if use_gpu and not any(key in index_name for key in keys_cpu):
        assert index_name in keys_cpu, "Unsupported Index for GPU"
        assert (
            faiss.StandardGpuResources
        ), "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
        params = faiss.GpuParameterSpace()
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        params = faiss.ParameterSpace()

    params.initialize(index)

    train_info = {
        "hw": "gpu" if use_gpu else "cpu",
        "dim": train_vec.shape[1],
        "num_sample": train_vec.shape[0],
    }
    print(f"training {index_name}...")
    t0 = time.time()
    index.train(train_vec)
    t1 = time.time()
    print(f"[{t1-t0}s] train, {index_name}")
    if save:
        if use_gpu:
            index_cpy = faiss.index_gpu_to_cpu(index)
        else:
            index_cpy = index
        faiss.write_index(index_cpy, save_path)
    return (index, params, train_info)


def add_idx(index, base_vec: NDArray):
    try:
        gindex = faiss.index_cpu_to_all_gpus(index)
        gindex.verbose = True
        gindex.add(base_vec)
        res_index = faiss.index_gpu_to_cpu(index)
    except Exception as e:
        print(e)
        print("Fallback to cpu...")
        index.verbose = True
        index.add(base_vec)
        res_index = index

    return res_index


def add_idx_dir_all(index_base_dir: str, index_output_dir: str, embs: NDArray):
    index_paths = [f for f in os.listdir(index_base_dir) if f.endswith(".faiss")]

    for index_filename in tqdm(index_paths, desc="Index"):
        output_path = os.path.join(index_output_dir, index_filename)
        index = faiss.read_index(index_filename)
        populated_idx = add_idx(index, embs)
        faiss.write_index(populated_idx, output_path)
        populated_idx = faiss.read_index(output_path)
        if populated_idx.ntotal != embs.shape[0]:
            print(output_path, "contains", populated_idx.ntotal, "/", embs.shape[0])
        else:
            print(output_path, " saved successfully")


def train_all(
    emb_path: str, train_ratio: float, dataset_prefix: str, train_info_path: str
):
    embs = load_embedding(emb_path)
    embs = random_sample_array(embs, train_ratio)
    dim = embs.shape[1]
    index_suffixes = get_index_list()
    train_data = []
    for index in tqdm(index_suffixes):
        train_info = {"index": index}
        try:
            _, _, info = train_idx(
                index,
                use_gpu=True,
                dim=dim,
                train_vec=embs,
                save=True,
                save_path=f"./{dataset_prefix}_{get_index_prefix(index)}.faiss",
            )
            train_info["detail"] = info
            train_data.append(train_info)
        except Exception as e:
            train_info["error"] = str(e)
            train_data.append(train_info)
    print("Train Summary=============")
    print(train_data)
    print("==========================")
    with open(train_info_path, "w") as f:
        json.dump(train_data, f, indent=4)


if __name__ == "__main__":
    EMB_PATH = "./data/emb.npy"
    train_all(EMB_PATH, 0.8, "wiki_1M", "./data/train_info.json")

    embs = load_embedding(EMB_PATH)
    add_idx_dir_all("./", "./data/index", embs)
