import os
from random import sample
import time
from datetime import datetime
from typing import List, Union, Tuple
from tqdm import tqdm
import datasets 
from datasets import load_dataset
from datasets.search import FaissIndex
import numpy as np
import cohere
import faiss
from dotenv import load_dotenv
from numpy.typing import NDArray
from multiprocessing import Pool, cpu_count
load_dotenv()

# TODO: 
# 1. dataset download
#   1-1. dataset load
#   1-2. dataset > ndarr conversion
# 2. faiss index type
# 3. index train
# 4. index add
# 5. retrieve
# 6. CLI arg parse

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

def save_ndarray(arr: NDArray, path: str):
    """
    Save ndarray as (.npy)
    """
    np.save(path, arr)

def process_chunk(args):
    ds, start, end, chunk_id, dtype = args
    batch = ds[start:end]
    batch_embs = np.array([row for row in batch['emb']], dtype=dtype)

    temp_file = f"temp_chunk_{chunk_id}.npy"
    np.save(temp_file, batch_embs)
    return temp_file

def dataset_2_numpy(ds):
    CHUNK_SIZE=100000
    OUTPUT_FILE = "wiki-en-emb.npy"
    NUM_PROC=cpu_count()-2
    num_rows = len(ds)
    dim = len(ds[0]['emb'])
    dtype = np.float32
    t0 = time.time()

    # 1. Numpy conversion iteration 1.5k/s
    # ds = load_dataset("Cohere/wikipedia-22-12-en-embeddings", cache_dir="./data", split='train', streaming=True)
    # embs = np.empty((num_rows, dim), dtype=dtype)
    # for i, row in tqdm(enumerate(ds)):
    #     embs[i] = np.array(row['emb'], dtype=dtype)
    # save_ndarray(embs, OUTPUT_FILE)

    # 2. Numpy conversion in chunked 
    # 1k --> 1.5k/s
    # 10k --> 58.7k/s
    # embs = np.memmap('embs.npy', dtype=dtype, mode='w+', shape=(num_rows, dim))
    # for i in tqdm(range(0, num_rows, CHUNK_SIZE)):
    #     batch = ds[i:i+CHUNK_SIZE]
    #     batch_embs = np.array([row for row in batch['emb']], dtype=dtype)
    #     embs[i:i+CHUNK_SIZE] = batch_embs
    # save_ndarray(embs, OUTPUT_FILE)

    # 3. Numpy conversion in multiprocess chunking
    # 30 process, 100K chunk 2815sec -> split
    print(f"Loaded dataset in {time.time()-t0:.2f}s with {num_rows} rows and {dim} dimensions.")
    print(f"Parallel processing strart w/ {NUM_PROC} process, {CHUNK_SIZE}chunk")
    chunk_args = [(ds, i, min(i + CHUNK_SIZE, num_rows), chunk_id, dtype)
              for chunk_id, i in enumerate(range(0, num_rows, CHUNK_SIZE))]
    t1 = time.time()
    # Parallel processing
    with Pool(NUM_PROC) as pool:
        temp_files = list(tqdm(pool.imap(process_chunk, chunk_args), total=len(chunk_args)))

    print(f"Parallel embedding processing took {time.time()-t1:.2f}s")

    # Merge temp files
    print("Merging temporary files into final np.memmap...")
    embs = np.memmap(OUTPUT_FILE, dtype=dtype, mode='w+', shape=(num_rows, dim))
    index = 0
    for temp_file in tqdm(temp_files):
        temp_data = np.load(temp_file, mmap_mode='r')
        embs[index: index + temp_data.shape[0]] = temp_data
        index += temp_data.shape[0]
        # os.remove(temp_file)
    del embs
    print(f"Embeddings saved to {OUTPUT_FILE} in {time.time()-t0:.2f}s")

    return OUTPUT_FILE

def download_dataset()-> (datasets.DatasetDict | datasets.Dataset | datasets.IterableDatasetDict | datasets.IterableDataset):
    t0 = time.time()
    ds = load_dataset("Cohere/wikipedia-22-12-en-embeddings", cache_dir="./data", split='train')
    print(f"Dataset load took {time.time()-t0}s")
    return ds

def load_dataset_ndarray(path: str)->NDArray:
    assert os.path.exists(path), f'{path} does not exists'
    return np.load(file=path, mmap_mode='r')
    
def get_embed(texts: List[str])-> NDArray:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)
    texts = ["hello world,", "what is the simplified explanation about notion of RAG"]
    # 768-dim
    response = co.embed(texts=texts, model="multilingual-22-12",)
    emb = response.embeddings
    emb = np.asarray(emb)
    print("emb shape:", emb.shape)
    return emb

def help_fn(msg, index: faiss.Index, params: Union[faiss.ParameterSpace, faiss.GpuParameterSpace]):
    print("Stage:",msg)
    print("Index cnt=", index.ntotal, "dim=", index.d, "trained=", index.is_trained)
    print("Params:")
    params.display()

def sample_train_vecs(dataset: NDArray, sample_ratio: float)-> NDArray:
    assert sample_ratio > 1 or sample_ratio < 0, "sample ratio must be between 0~1"
    num_samples = int(sample_ratio * dataset.shape[0])
    indicies = np.random.choice(dataset.shape[0], num_samples, replace=False)
    return dataset[indicies]
 
def train_idx(index_name: str, use_gpu: bool, dim: int, train_vec: NDArray, save: bool, save_path: Union[None, str])->Tuple[faiss.Index, faiss.ParameterSpace, NDArray]:
    index = faiss.index_factory(dim, index_name)

    if use_gpu:
        keys_gpu = [
            "PCA64,IVF4096,Flat",
            "PCA64,Flat", "Flat", "IVF4096,Flat", "IVF16384,Flat",
            "IVF4096,PQ32"]
        assert index_name in keys_gpu, "Unsupported Index for GPU"
        assert faiss.StandardGpuResources, \
            "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
        res = faiss.StandardGpuResources()
        dev_no = 0
        params = faiss.GpuParameterSpace()
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        params = faiss.ParameterSpace()

    params.initialize(index)

    help_fn("training...", index, params)
    t0 = time.time()
    index.train(train_vec)
    print("[%.3f s] train", time.time()-t0)
    print("train index:", index_name)
    print("train hw:", "gpu" if use_gpu else "cpu")
    print("train dataset:", train_vec.shape)
    if save:
        if use_gpu:
            index_cpy = faiss.index_gpu_to_cpu(index)
        else:
            index_cpy = index
        faiss.write_index(index_cpy, f'./train_idx_{index_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    return (index, params, train_vec)

def add_idx(index: faiss.IndexPreTransform, base_vec: NDArray) -> faiss.Index:
    index.add(n=base_vec.shape[0], x=base_vec)
    return index

def retrieve(query_texts: List[str], k: int, index, dataset):
    query_embeddings = get_embed(query_texts)

    # Search
    distances, indices = index.search(query_embeddings.shape[0], query_embeddings, k)

    results = []
    for q_idx, query in enumerate(query_texts):
        query_results = []
        for i, doc_idx in enumerate(indices[q_idx]):
            retrieved_doc = dataset[doc_idx]  # Retrieve the document
            query_results.append({
                "query": query,
                "retrieved_title": retrieved_doc["title"],
                "retrieved_text": retrieved_doc["text"],
                "distance": distances[q_idx][i]
            })
        results.append(query_results)

    return results

if __name__ == "__main__":
    ds = download_dataset()
    emb_file = dataset_2_numpy(ds)
    # emb_file = "wiki-en-emb.npy"
    train_ratio = 0.6
    emb = load_dataset_ndarray(emb_file)
    print("Whole embedding ", emb.shape)
    train_emb = sample_train_vecs(emb, train_ratio)
    print("Training embedding", train_emb.shape)

    print("Done!")
