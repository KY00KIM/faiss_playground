from argparse import Namespace
import time
import os

from pathlib import Path
from typing import Dict, List
import multiprocessing
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


# def process_chunk(args):
#     ds, start, end, chunk_id, dtype = args
#     batch = ds[start:end]
#     batch_embs = np.array([row for row in batch["emb"]], dtype=dtype)
#
#     temp_file = f"temp_chunk_{chunk_id}.npy"
#     np.save(temp_file, batch_embs)
#     return temp_file
#
#
# def merge_temp_npy(temp_files: List[str], output_file: str, dtype, num_rows, dim):
#     t0 = time.time()
#     print("Merging temporary files into final np.memmap...")
#     embs = np.memmap(output_file, dtype=dtype, mode="w+", shape=(num_rows, dim))
#     index = 0
#     for temp_file in tqdm(temp_files):
#         temp_data = np.load(temp_file, mmap_mode="r")
#         embs[index : index + temp_data.shape[0]] = temp_data
#         index += temp_data.shape[0]
#     del embs
#     print(f"Embeddings saved to {output_file} in {time.time()-t0:.2f}s")
#
#     return output_file


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
    ds = load_dataset(dataset_path, cache_dir=cache_dir, split=split)
    print(f"Dataset load took {time.time()-t0}s")
    return ds


def convert_chunk_to_ndarr(
    ds,
    batched: bool,
    batch_size: int | None = None,
    num_proc: int | None = None,
) -> NDArray:
    def concat_title_batch(batch: Dict[str, List]) -> Dict[str, List]:
        for i in range(len(batch)):
            batch["text"][i] = batch["title"][i] + " " + batch["text"][i]
        return batch

    t0 = time.time()
    ds = ds.map(
        concat_title_batch, batched=batched, batch_size=batch_size, num_proc=num_proc
    )
    print("Mapping took", time.time() - t0, "sec")
    for row in range(5):
        print(ds[row]["title"], ":", ds[row]["text"])
    text_list = ds["text"]

    t0 = time.time()
    arr = np.array(text_list, dtype=object)
    print("ndarr conversion took", time.time() - t0, "sec")
    return arr


def extract_embedding(dataset, dtype: DTypeLike, chunk_size: int = 1000) -> NDArray:
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

    return embs


def main(args: Namespace):
    ds = download_dataset(args.dataset, args.dataset_cache_dir, args.dataset_split)

    if args.process == "all" or args.process == "extract-emb":
        assert hasattr(
            args, "output_emb"
        ), "output path for embedding is not provided (--output-emb)"
        os.makedirs(os.path.dirname(args.output_emb), exist_ok=True)
        embs = extract_embedding(ds, np.float32, chunk_size=args.batch)
        np.save(args.output_emb, embs)
        try:
            embs = np.load(args.output_emb, mmap_mode="r")
            print("Successfully loaded after save:", embs.shape)
        except Exception as e:
            print("Failed to load after save:", e)

    if args.process == "all" or args.process == "extract-chunk":
        assert hasattr(
            args, "output_chunk"
        ), "output path for embedding is not provided (--output-emb)"
        assert (
            args.chunk_cpu_ratio <= 1 and args.chunk_cpu_ratio > 0
        ), "cpu utilization ratio must be between 0 and 1 (--chunk-cpu-ratio)"
        chunks = convert_chunk_to_ndarr(
            ds,
            batched=True,
            batch_size=args.batch,
            num_proc=max(1, int(multiprocessing.cpu_count() * args.chunk_cpu_ratio)),
        )
        os.makedirs(os.path.dirname(args.output_chunk), exist_ok=True)
        np.save(args.output_chunk, chunks)
        try:
            del chunks
            chunks = np.load(args.output_chunk, mmap_mode="r", allow_pickle=True)
            print("Successfully loaded after save:", chunks.shape)
        except Exception as e:
            print("Failed to load after save:", e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument(
        "--process",
        type=str,
        required=True,
        choices=["extract-emb", "extract-chunk", "all"],
    )
    parser.add_argument("--output-emb", type=Path)
    parser.add_argument("--output-chunk", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-cpu-ratio", type=float, default=1)
    args = parser.parse_args()

    # print("PROVIDED ARGS===============")
    # print(args)
    # print("============================")

    main(args)

    DATASET_HF_NAME = "Cohere/wikipedia-22-12-en-embeddings"
    HF_CACHE_DIR = "/mnt/raid0/repo/vecdb_data"
    HF_DATASET_SPLIT = "train"
    EMB_PATH = "/mnt/raid0/repo/vecdb_data/emb.npy"
