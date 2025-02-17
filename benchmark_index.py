import os
import time
import resource
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import faiss
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm


@dataclass
class PerfCounters:
    wall_time_s: float = 0.0
    user_time_s: float = 0.0
    system_time_s: float = 0.0
    peak_mem_diff_kb: float = 0.0
    peak_mem_kb: float = 0.0


@contextmanager
def timed_execution() -> Generator[PerfCounters, None, None]:
    pcounters = PerfCounters()
    wall_time_start = time.perf_counter()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)
    yield pcounters
    wall_time_end = time.perf_counter()
    rusage_end = resource.getrusage(resource.RUSAGE_SELF)
    pcounters.wall_time_s = wall_time_end - wall_time_start
    pcounters.user_time_s = rusage_end.ru_utime - rusage_start.ru_utime
    pcounters.system_time_s = rusage_end.ru_stime - rusage_start.ru_stime
    pcounters.peak_mem_diff_kb = rusage_end.ru_maxrss - rusage_start.ru_maxrss
    pcounters.peak_mem_kb = rusage_end.ru_maxrss


def accumulate_perf_counter(
    t: PerfCounters,
):
    KB_IN_MB = 1_024
    counters = {}
    counters["wall_time_s"] = t.wall_time_s
    counters["user_time_s"] = t.user_time_s
    counters["peak_mem_diff_mb"] = t.peak_mem_diff_kb / KB_IN_MB
    counters["peak_mem_mb"] = t.peak_mem_kb / KB_IN_MB
    return counters


def new_benchmark_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "index_name",
            "top_k",
            "nprobe",
            "efSearch",
            "batch_size",
            "cpu_nthreads",
            "hw",
            "peak_mem_mb",
            "peak_mem_diff_mb",
            # "user_time_s",
            "wall_time_ms",
        ]
    )


def append_benchmark_df(
    df: pd.DataFrame,
    index_name: str,
    top_k: int,
    nprobe: None | int,
    efSearch: None | int,
    batch_size: int,
    use_gpu: bool,
    cpu_nthreads: int,
    **kwargs,
) -> pd.DataFrame:
    perf_counter = {
        "index_name": index_name.split("wiki_1M_")[-1],
        "top_k": top_k,
        "nprobe": nprobe,
        "efSearch": efSearch,
        "batch_size": batch_size,
        "cpu_nthreads": None if use_gpu else cpu_nthreads,
        "hw": "gpu(1)" if use_gpu else f"cpu({faiss.omp_get_max_threads()})",
        "peak_mem_mb": kwargs.get("peak_mem_mb", None),
        "peak_mem_diff_mb": kwargs.get("peak_mem_diff_mb", None),
        # "user_time_s": kwargs.get("user_time_s", None),
        "wall_time_ms": kwargs.get("wall_time_s", None),
    }
    if perf_counter["wall_time_ms"] is not None:
        perf_counter["wall_time_ms"] = perf_counter["wall_time_ms"] * 1000
    new_row = pd.DataFrame([perf_counter])
    df = pd.concat([df, new_row], ignore_index=True)

    return df


def benchmark_index(
    index,
    embs: NDArray,
    index_prefix: str,
    perf_df: pd.DataFrame,
    max_batch: int = 128,
    use_gpu: bool = False,
):
    emb_mean = np.mean(embs, axis=0)
    emb_std = np.std(embs, axis=0)
    dim = embs.shape[1]
    batch_size = 1
    ps = faiss.ParameterSpace()
    max_threads = faiss.omp_get_max_threads()
    exec_threads = faiss.omp_get_max_threads()
    min_thread = 8
    while exec_threads >= min_thread:
        print(exec_threads, "thread")
        if not use_gpu:
            faiss.omp_set_num_threads(exec_threads)
        for k in range(1, 5):
            if "IVF" in index_prefix:
                # IVF based search
                for nprobe in 1, 4, 16, 64, 256:
                    batch_size = 1
                    if use_gpu:
                        index = faiss.index_gpu_to_cpu(index)
                    ps.set_index_parameter(index, "nprobe", nprobe)
                    if use_gpu:
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                        # index = faiss.index_cpu_to_all_gpus(index)
                    while batch_size <= max_batch:
                        sample_batch = np.random.normal(
                            emb_mean, emb_std, size=(batch_size, dim)
                        )
                        with timed_execution() as t:
                            distances, indices = index.search(sample_batch, k)
                        t = accumulate_perf_counter(t)
                        perf_df = append_benchmark_df(
                            df=perf_df,
                            index_name=index_prefix,
                            top_k=k,
                            nprobe=nprobe,
                            efSearch=None,
                            batch_size=batch_size,
                            use_gpu=use_gpu,
                            cpu_nthreads=exec_threads,
                            **t,
                        )
                        batch_size = batch_size * 2
            elif "HNSW" in index_prefix:
                # graph-based search
                for hnsw_search_window in 16, 32, 64, 128, 256:
                    batch_size = 1
                    if use_gpu:
                        index = faiss.index_gpu_to_cpu(index)
                    # in case of opaque index w/ pretransform
                    if not hasattr(index, "hnsw"):
                        sub_index = faiss.downcast_index(index.index)
                        sub_index.hnsw.efSearch = hnsw_search_window
                    # in case of index w/o pretransform
                    else:
                        index.hnsw.efSearch = hnsw_search_window
                    if use_gpu:
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                        # index = faiss.index_cpu_to_all_gpus(index)
                    while batch_size <= max_batch:
                        sample_batch = np.random.normal(
                            emb_mean, emb_std, size=(batch_size, dim)
                        )
                        with timed_execution() as t:
                            distances, indices = index.search(sample_batch, k)
                        t = accumulate_perf_counter(t)
                        perf_df = append_benchmark_df(
                            df=perf_df,
                            index_name=index_prefix,
                            top_k=k,
                            nprobe=None,
                            efSearch=hnsw_search_window,
                            batch_size=batch_size,
                            use_gpu=use_gpu,
                            cpu_nthreads=exec_threads,
                            **t,
                        )
                        batch_size = batch_size * 2
            else:
                # exhaustive search
                batch_size = 1
                while batch_size <= max_batch:
                    sample_batch = np.random.normal(
                        emb_mean, emb_std, size=(batch_size, dim)
                    )
                    with timed_execution() as t:
                        distances, indices = index.search(sample_batch, k)
                    t = accumulate_perf_counter(t)
                    perf_df = append_benchmark_df(
                        df=perf_df,
                        index_name=index_prefix,
                        top_k=k,
                        nprobe=None,
                        efSearch=None,
                        batch_size=batch_size,
                        use_gpu=use_gpu,
                        cpu_nthreads=exec_threads,
                        **t,
                    )
                    batch_size = batch_size * 2
        if use_gpu:
            break
        else:
            exec_threads = exec_threads // 2
    faiss.omp_set_num_threads(max_threads)
    print(perf_df.describe())
    print(perf_df.tail())
    return perf_df


def benchmark_all_index(
    idx_dir: str,
    embs: NDArray,
    perf_df: pd.DataFrame,
    perf_df_save_path: str,
):
    # Process w/ BLAS from batch size 1 (default 20)
    faiss.cvar.distance_compute_blas_threshold = 1
    for index_filename in tqdm(
        [file for file in os.listdir(idx_dir) if file.endswith(".faiss")]
    ):
        max_batch = 256
        index_name = index_filename.split(".")[0]
        index_path = os.path.join(idx_dir, index_filename)
        index = faiss.read_index(index_path)
        print(index_name, "starting cpu benchmark..")
        perf_df = benchmark_index(
            index,
            embs,
            index_name,
            perf_df,
            max_batch,
            False,
        )
        try:
            print(index_name, "starting gpu benchmark..")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            # gpu_index = faiss.index_cpu_to_all_gpus(index)
            perf_df = benchmark_index(
                gpu_index,
                embs,
                index_name,
                perf_df,
                max_batch,
                True,
            )
        except Exception as e:
            print(e)
            print(index_name, "as gpu benchmark failed")
            continue
    perf_df.to_csv(perf_df_save_path)


if __name__ == "__main__":
    DATASET_PATH = "./wikipedia-22-12-en-embeddings_1M"
    emb_path = "wiki-1M-emb.npy"
    dataset_prefix = "wiki_1M"
    train_info_output = "train_data.json"
    index_path = "wiki_1M_Flat.faiss"
    train_ratio = 0.8
    dim = 768

    df = new_benchmark_df()
    embs = np.load("wiki-1M-emb.npy", mmap_mode="r")
    benchmark_all_index("./data/index", embs, df, "./data/benchmark/benchmark.csv")
