from typing import List
import cohere
import os
from numpy.typing import NDArray
import numpy as np


def get_index_prefix(index_name: str) -> str:
    return index_name.replace(",", "_")


def get_index_list() -> List[str]:
    def concat_index_name(transform: str, search: str, encoding: str) -> str:
        res = ""
        if transform != "None":
            res += transform + ","
        if search != "None":
            res += search + ","
        res += encoding
        return res

    vec_transform_suffix = ["None", "PCA64", "OPQ16_64"]
    search_suffix = [
        "None",
        "IVF1024",
        "IVF4096",
        "IVF65536",  # recommend for 2M<
        "IVF65536_HNSW32",  # error
        "HNSW32",
    ]
    encoding_suffix = [
        "Flat",
        "PQ16",
    ]
    res = []
    for transform in vec_transform_suffix:
        for search in search_suffix:
            for encoding in encoding_suffix:
                res.append(concat_index_name(transform, search, encoding))
                print(concat_index_name(transform, search, encoding))
        print()
    return res


def get_embed(texts: List[str]) -> NDArray:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(cohere_api_key)
    # texts = ["hello world,", "what is the simplified explanation about notion of RAG"]
    # 768-dim
    response = co.embed(
        texts=texts,
        model="multilingual-22-12",
    )
    emb = response.embeddings
    emb = np.asarray(emb)
    print("emb shape:", emb.shape)
    return emb


def save_ndarray(arr: NDArray, path: str):
    """
    Save ndarray as (.npy)
    """
    np.save(path, arr)


def load_embedding(path: str) -> NDArray:
    assert os.path.exists(path), f"{path} does not exists"
    return np.load(file=path, mmap_mode="r")
