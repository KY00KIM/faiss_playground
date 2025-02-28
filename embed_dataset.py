import multiprocessing
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import datasets
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from numpy.typing import DTypeLike, NDArray
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


def extract_embedding_mmap(
    dataset, memmap_f: str, dtype: DTypeLike, chunk_size: int = 1000
):
    """
    Assumption:
        - embedding at `dataset['emb']`
        - embedding at all same dimension
    """
    ntotal = len(dataset)
    dim = len(dataset[0]["emb"])

    embs_mem = np.memmap(
        memmap_f,
        dtype=dtype,
        mode="w+",
        shape=(ntotal, dim),
    )

    for start_idx in tqdm(range(0, ntotal, chunk_size)):
        end_idx = min(start_idx + chunk_size, ntotal)
        batch = dataset[start_idx:end_idx]
        batch_embs = np.stack(batch["emb"]).astype(dtype)
        embs_mem[start_idx:end_idx, :] = batch_embs

    embs_mem.flush()
    np.save(memmap_f.split(".memmap")[0], embs_mem)
    del embs_mem
    os.remove(memmap_f)


def main(args: Namespace):
    ds = download_dataset(args.dataset, args.dataset_cache_dir, args.dataset_split)

    if args.process == "all" or args.process == "extract-emb":
        assert hasattr(
            args, "output_emb"
        ), "output path for embedding is not provided (--output-emb)"
        os.makedirs(os.path.dirname(args.output_emb), exist_ok=True)
        extract_embedding_mmap(
            ds, args.output_emb + ".memmap", np.float32, chunk_size=args.batch
        )
        try:
            embs = np.memmap(
                args.output_emb,
                dtype=np.float32,
                mode="r",
                shape=(len(ds), len(ds[0]["emb"])),
            )
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
