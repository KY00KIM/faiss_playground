import os
import time
from typing import List, Union, Tuple
from datasets import load_dataset
from datasets.search import FaissIndex
import numpy as np
import cohere
import faiss
from dotenv import load_dotenv
from numpy.typing import NDArray
load_dotenv()

# TODO: 
# 1. dataset download
# 2. faiss index type
# 3. index train
# 4. index add
# 5. retrieve
# 6. CLI arg parse

def get_dataset():
    ds = load_dataset("Cohere/wikipedia-22-12-en-embeddings", cache_dir="./data", split="train")

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

def train_idx(index_name: str, use_gpu: bool, dim: int, train_vec: NDArray)->Tuple[faiss.Index, faiss.ParameterSpace, NDArray]:
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
    return (index, params, train_vec)

def add_idx(index: faiss.IndexPreTransform, base_vec: NDArray) -> faiss.Index:
    index.add(n=base_vec.shape[0], x=base_vec)
    return index

# TODO: impl retrieval wrt piperag
# def retrieve(query_embedding: NDArray, k: int=5, index: faiss.Index):
#     # Ensure the query embedding is a 2D array
#     query_embedding = np.array(query_embedding).reshape(1, -1)
#
#     # Search the index for the k nearest neighbors
#     distances, indices = index.search(n=query_embedding.shape[0], x=query_embedding, k=k)
#
#     # Retrieve the corresponding titles and texts
#     results = [{"title": titles[i], "text": texts[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
#     return results

if __name__ == "__main__":
    get_dataset()
