import time
from typing import List

import datasets
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from numpy.typing import NDArray, DTypeLike
from tqdm import tqdm

load_dotenv()


def print_dataset_info(dataset):
    print("\n Dataset Information:")
    print(f"- Dataset Type: {type(dataset)}")

    if isinstance(dataset, dict):
        print(f"- Available Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"\nSplit: {split_name}")
            print(f"  - Number of Samples: {len(split_data)}")
            print(f"  - Columns: {split_data.column_names}")
            print(f"  - Data Types: {split_data.features}")
    else:
        print(f"- Number of Samples: {len(dataset)}")
        print(f"- Columns: {dataset.column_names}")
        print(f"- Data Types: {dataset.features}")


def process_chunk(args):
    ds, start, end, chunk_id, dtype = args
    batch = ds[start:end]
    batch_embs = np.array([row for row in batch["emb"]], dtype=dtype)

    temp_file = f"temp_chunk_{chunk_id}.npy"
    np.save(temp_file, batch_embs)
    return temp_file


def merge_temp_npy(temp_files: List[str], output_file: str, dtype, num_rows, dim):
    t0 = time.time()
    print("Merging temporary files into final np.memmap...")
    embs = np.memmap(output_file, dtype=dtype, mode="w+", shape=(num_rows, dim))
    index = 0
    for temp_file in tqdm(temp_files):
        temp_data = np.load(temp_file, mmap_mode="r")
        embs[index : index + temp_data.shape[0]] = temp_data
        index += temp_data.shape[0]
    del embs
    print(f"Embeddings saved to {output_file} in {time.time()-t0:.2f}s")

    return output_file


# def dataset_2_numpy(ds):
#     MAX_DOC = 10_000_000
#     CHUNK_SIZE = 100_000
#     OUTPUT_FILE = "wiki-en-emb.npy"
#     NUM_PROC = cpu_count() - 2
#     num_rows = len(ds)
#     dim = len(ds[0]["emb"])
#     dtype = np.float32
#
#     # 1. Numpy conversion iteration 1.5k/s
#     # ds = load_dataset("Cohere/wikipedia-22-12-en-embeddings", cache_dir="./data", split='train', streaming=True)
#     # embs = np.empty((num_rows, dim), dtype=dtype)
#     # for i, row in tqdm(enumerate(ds)):
#     #     embs[i] = np.array(row['emb'], dtype=dtype)
#     # save_ndarray(embs, OUTPUT_FILE)
#
#     # 2. Numpy conversion in chunked
#     # 1k --> 1.5k/s
#     # embs = np.memmap('embs.npy', dtype=dtype, mode='w+', shape=(num_rows, dim))
#     # for i in tqdm(range(0, num_rows, CHUNK_SIZE)):
#     #     batch = ds[i:i+CHUNK_SIZE]
#     #     batch_embs = np.array([row for row in batch['emb']], dtype=dtype)
#     #     embs[i:i+CHUNK_SIZE] = batch_embs
#     # save_ndarray(embs, OUTPUT_FILE)
#
#     # 3. Numpy conversion in multiprocess chunking
#     # 30 process, 100K chunk 2815sec -> split
#     print(f"Parallel processing strart w/ {NUM_PROC} process, {CHUNK_SIZE}chunk")
#     chunk_args = [
#         (ds, i, min(i + CHUNK_SIZE, num_rows), chunk_id, dtype)
#         for chunk_id, i in enumerate(range(0, num_rows, CHUNK_SIZE))
#     ]
#     # multiproc processing
#     t1 = time.time()
#     with Pool(NUM_PROC) as pool:
#         temp_files = list(
#             tqdm(pool.imap(process_chunk, chunk_args), total=len(chunk_args))
#         )
#     print(f"Parallel embedding processing took {time.time()-t1:.2f}s")
#
#     # Merge temp files
#     temp_files = [
#         f"temp_chunk_{chunk_id}.npy"
#         for chunk_id, _ in enumerate(range(0, int(num_rows * 0.5), CHUNK_SIZE))
#     ]
#     merge_temp_npy(temp_files, OUTPUT_FILE, dtype, num_rows, CHUNK_SIZE)
#
#     return OUTPUT_FILE


def download_dataset(
    dataset_path: str, cache_dir: str, split="train"
) -> (
    datasets.DatasetDict
    | datasets.Dataset
    | datasets.IterableDatasetDict
    | datasets.IterableDataset
):
    t0 = time.time()
    ds = load_dataset(dataset_path, cache_dir, split)
    print(f"Dataset load took {time.time()-t0}s")
    return ds


def split_dataset(ds, output_path: str, output_size: int, strategy: str = "default"):
    if strategy == "shuffle":
        ds = ds.shuffle(seed=1234)
    split_ds = ds.select(range(output_size))
    split_ds.save_to_disk(output_path)


def extract_embedding(
    dataset, dtype: DTypeLike, save_path: str | None, chunk_size: int = 1000
) -> NDArray:
    """
    Assumption:
        - embedding at `dataset['emb']`
        - embedding at all same dimension
    """
    ntotal = len(dataset)
    dim = len(dataset[0]["emb"])
    embs = np.empty(shape=(ntotal, dim), dtype=dtype)

    for i in tqdm(range(0, ntotal, chunk_size)):
        batch = dataset[i : i + chunk_size]
        batch_embs = np.stack(batch["emb"]).astype(dtype)
        embs[i : i + chunk_size] = batch_embs

    if save_path is not None:
        np.save(save_path, embs)
        try:
            embs = np.load(save_path, mmap_mode="r")
            print("Successfully loaded after save:", embs.shape)
        except Exception as e:
            print("Failed to load after save:", e)
    return embs


if __name__ == "__main__":
    DATASET_HF_NAME = "Cohere/wikipedia-22-12-en-embeddings"
    HF_CACHE_DIR = "./data"
    HF_DATASET_SPLIT = "train"
    EMB_PATH = "./data/emb.npy"
    ds = download_dataset(DATASET_HF_NAME, HF_CACHE_DIR, HF_DATASET_SPLIT)
    emb = extract_embedding(ds, np.float32, EMB_PATH)
