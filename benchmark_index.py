import os
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

import faiss


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
            "recall_1",
            "ms_per_query",
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
        "recall_1": kwargs.get("recall_1", None),
        "ms_per_query": kwargs.get("sec_per_query", None),
    }
    if perf_counter["wall_time_ms"] is not None:
        perf_counter["wall_time_ms"] = perf_counter["wall_time_ms"] * 1000
    if perf_counter["ms_per_query"] is not None:
        perf_counter["ms_per_query"] = perf_counter["ms_per_query"] * 1000
    new_row = pd.DataFrame([perf_counter])
    df = pd.concat([df, new_row], ignore_index=True)

    return df


def benchmark_index(
    index,
    query: NDArray,
    gt: NDArray,
    index_prefix: str,
    perf_df: pd.DataFrame,
    use_gpu: bool = False,
):
    n_gt = gt.shape[0]
    batch_size = 1
    ps = faiss.ParameterSpace()
    max_threads = faiss.omp_get_max_threads()
    exec_threads = faiss.omp_get_max_threads()
    min_thread = 16
    k = 1
    while exec_threads >= min_thread:
        print(exec_threads, "thread")
        if not use_gpu:
            faiss.omp_set_num_threads(exec_threads)
        # for k in range(1, 5):
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
                # while batch_size <= max_batch:
                with timed_execution() as t:
                    dists, ids = index.search(query, k)
                t = accumulate_perf_counter(t)
                res = 0
                for i in range(n_gt):
                    if gt[i] in ids[i, :]:
                        res += 1
                recall = res / n_gt
                t[f"recall_{k}"] = recall
                t["sec_per_query"] = t["wall_time_s"] / n_gt
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
                # batch_size = batch_size * 2
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
                # while batch_size <= max_batch:
                with timed_execution() as t:
                    dists, ids = index.search(query, k)
                t = accumulate_perf_counter(t)
                res = 0
                for i in range(n_gt):
                    if gt[i] in ids[i, :]:
                        res += 1
                recall = res / n_gt
                t[f"recall_{k}"] = recall
                t["sec_per_query"] = t["wall_time_s"] / n_gt
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
                # batch_size = batch_size * 2
        else:
            # exhaustive search
            batch_size = 1
            with timed_execution() as t:
                dists, ids = index.search(query, k)
            t = accumulate_perf_counter(t)
            res = 0
            for i in range(n_gt):
                if gt[i] in ids[i, :]:
                    res += 1
            recall = res / n_gt
            t[f"recall_{k}"] = recall
            t["sec_per_query"] = t["wall_time_s"] / n_gt
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
            # batch_size = batch_size * 2
        if use_gpu:
            break
        else:
            exec_threads = exec_threads // 2
    faiss.omp_set_num_threads(max_threads)
    print(perf_df.describe())
    print(perf_df.tail())
    return perf_df


def create_query_gt(flat_idx, embs: NDArray, size: int, k: int):
    query = embs[len(embs) - size :]
    _, gt = flat_idx.search(query, k)
    return (query, gt)


def benchmark_all_index(
    idx_dir: str,
    embs: NDArray,
    perf_df: pd.DataFrame,
    perf_df_save_path: str,
    query_size: int = 1000,
):
    # Process w/ BLAS from batch size 1 (default 20)
    faiss.cvar.distance_compute_blas_threshold = 1
    flat_file_name = [
        file
        for file in os.listdir(idx_dir)
        if file.endswith(".faiss")
        and "Flat" in file
        and "IVF" not in file
        and "PQ" not in file
    ]
    assert len(flat_file_name) != 0, f"Flat index is not in index dir {idx_dir}"
    flat_file_name = flat_file_name[0]
    query, gt = create_query_gt(
        faiss.read_index(os.path.join(idx_dir, flat_file_name)), embs, query_size, 1
    )
    gt = gt.reshape(-1, 1)
    print("Query shape:", query.shape, "GT shape:", gt.shape)
    for i, index_filename in tqdm(
        enumerate([file for file in os.listdir(idx_dir) if file.endswith(".faiss")])
    ):
        index_name = index_filename.split(".")[0]
        if (
            os.path.exists(perf_df_save_path)
            and (perf_df["index_name"] == index_name).any()
        ):
            print(index_name, "exists in ", perf_df_save_path, "passing...")
            continue
        print(
            f"benchmark start {idx_dir}/{index_filename}({i+1}/{len(os.listdir(idx_dir))})"
        )
        index_path = os.path.join(idx_dir, index_filename)
        index = faiss.read_index(index_path)
        query_size = 1000
        print(index_name, "starting cpu benchmark..")
        perf_df = benchmark_index(
            index=index,
            query=query,
            gt=gt,
            index_prefix=index_name,
            perf_df=perf_df,
            use_gpu=False,
        )
        try:
            print(index_name, "starting gpu benchmark..")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            perf_df = benchmark_index(
                index=gpu_index,
                query=query,
                gt=gt,
                index_prefix=index_name,
                perf_df=perf_df,
                use_gpu=True,
            )
        except Exception as e:
            print(e)
            print(index_name, "as gpu benchmark failed")
            with open(
                os.path.join(os.path.dirname(perf_df_save_path), "gpu_error.txt"), "a"
            ) as f:
                f.write(f"{index_name} Index GPU Benchmark Error\n{e}\n")
            continue
        del index
        perf_df.to_csv(perf_df_save_path)


def plot_OperatingPoints(ops, nq, **kwargs):
    ops = ops.optimal_pts
    n = ops.size() * 2 - 1
    plt.plot(
        [ops.at(i // 2).perf for i in range(n)],
        [ops.at((i + 1) // 2).t / nq * 1000 for i in range(n)],
        **kwargs,
    )


def save_plot(index_name: str, hw: str, output_dir: str, op_per_key, crit):
    fig = plt.figure(figsize=(12, 9))
    plt.xlabel("Recall at 1")
    plt.ylabel("search time (ms/query, %d threads)" % faiss.omp_get_max_threads())
    plt.gca().set_yscale("log")
    plt.title(f"{index_name} {hw}")
    plt.grid()
    res = {}
    for i2, opi2 in op_per_key:
        plot_OperatingPoints(opi2, crit.nq, label=i2, marker="o")
        ops = opi2.optimal_pts
        n = ops.size() * 2 - 1
        res[index_name] = [
            (ops.at(i // 2).perf, ops.at((i + 1) // 2).t / crit.nq * 1000)
            for i in range(n)
        ]
    plt.legend(loc=2)
    fig.savefig(os.path.join(output_dir, "auto_tune" + index_name + "_" + hw + ".png"))
    print(
        f"saved: {os.path.join(output_dir, 'auto_tune' + index_name + '_' + hw +'.png')}"
    )
    return res


def faiss_auto_tune(
    idx_dir: str,
    embs: NDArray,
    output_dir: str,
    query_size: int = 1000,
):
    # Process w/ BLAS from batch size 1 (default 20)
    faiss.cvar.distance_compute_blas_threshold = 1
    os.makedirs(output_dir, exist_ok=True)
    flat_file_name = [
        file
        for file in os.listdir(idx_dir)
        if file.endswith(".faiss")
        and "Flat" in file
        and "IVF" not in file
        and "PQ" not in file
    ]
    assert len(flat_file_name) != 0, f"Flat index is not in index dir {idx_dir}"
    flat_file_name = flat_file_name[0]
    query, gt = create_query_gt(
        faiss.read_index(os.path.join(idx_dir, flat_file_name)), embs, query_size, 1
    )

    for i, index_filename in enumerate(
        [file for file in os.listdir(idx_dir) if file.endswith(".faiss")]
    ):
        print(f"auto_tune start {idx_dir}/{index_filename}")
        t0 = time.perf_counter()
        index_name = index_filename.split(".")[0]

        index = faiss.read_index(os.path.join(idx_dir, index_filename))
        index.verbose = True

        crit = faiss.OneRecallAtRCriterion(query.shape[0], 1)
        crit.set_groundtruth(None, gt)
        crit.nnn = 1
        op = faiss.OperatingPoints()
        print("set crit")
        hw = f"cpu({faiss.omp_get_max_threads()})"
        params = faiss.ParameterSpace()
        params.verbose = True
        params.initialize(index)
        if os.path.exists(
            os.path.join(output_dir, "auto_tune" + index_name + "_" + hw + ".png")
        ):
            continue

        print("exploring params")
        t0 = time.perf_counter()
        opi = params.explore(index, query, crit)
        print("explored params", time.perf_counter() - t0, "sec")
        opi.display()

        op.merge_with(opi, index_name + " ")
        op_per_key = []
        op_per_key.append((index_name, opi))

        print("saving output")
        save_plot(
            index_name=index_name,
            hw=hw,
            output_dir=output_dir,
            op_per_key=op_per_key,
            crit=crit,
        )
        print("saved output")
        t1 = time.perf_counter()


def main(args):
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    # embs = np.memmap(args.emb_path, mode="r", dtype=np.float32, shape=(35167920, 768))
    embs = np.load(args.emb_path, mmap_mode="r")
    if args.process == "benchmark":
        if os.path.exists(args.output_csv):
            df = pd.read_csv(args.output_csv)
        else:
            df = new_benchmark_df()
        benchmark_all_index(
            idx_dir=args.index_dir,
            embs=embs,
            perf_df=df,
            perf_df_save_path=args.output_csv,
            query_size=args.query_size,
        )
    if args.process == "auto_tune":
        faiss_auto_tune(
            idx_dir=args.index_dir,
            embs=embs,
            output_dir=os.path.dirname(args.output_csv),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process",
        type=str,
        default="benchmark",
        choices=["benchmark", "auto_tune"],
        # required=True,
        help="Directory containing .faiss index files",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        # default="./data/index",
        required=True,
        help="Directory containing .faiss index files",
    )
    parser.add_argument(
        "--emb-path",
        type=str,
        # default="emb.npy",
        required=True,
        help="Numpy embedding file to load",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        # default="./data/benchmark/benchmark.csv",
        required=True,
        help="Path to save final results csv",
    )
    parser.add_argument(
        "--quey-size",
        type=int,
        default=1000,
        help="Query size for evaluation",
    )
    args = parser.parse_args()

    main(args)
