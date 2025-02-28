# -----------------------------------------------------------
# Variables: `make extract-emb DATASET=/path/to/dataset`
# -----------------------------------------------------------
# -----------------------------------------------------------
# Dataset Extraction Variables:
# -----------------------------------------------------------
DATASET         ?= "Cohere/wikipedia-22-12-en-embeddings"
DATASET_CACHE_DIR   ?= "~/.cache/huggingface/hub"
DATASET_SPLIT   ?= train
OUTPUT_EMB      ?= wiki-1M-emb.npy
OUTPUT_CHUNK    ?= wiki-1M-chunk.npy
INDEX_TYPE      ?= Flat
PROCESS         ?= extract-emb
CHUNK_CPU_RATIO ?= 1
BATCH           ?= 1000

# -----------------------------------------------------------
# Index Variables:
# -----------------------------------------------------------
INDEX            ?= Flat IVF1024,Flat IVF1024,PQ16 IVF1024,PQ64 IVF4096,Flat IVF4096,PQ16 IVF4096,PQ64 IVF65536,Flat IVF65536,PQ16 
INDEX_BASE_DIR   ?= ./index_base
INDEX_OUTPUT_DIR ?= ./index_populated
EMBEDDING_PATH   ?= ./wiki-1M-emb.npy
EMBEDDING_NUM    ?= 1000000
TRAIN_RATIO      ?= 0.8
OUTPUT_PREFIX    ?= wiki_1M
TRAIN_JSONL_PATH ?= train_data.jsonl
NUM_GPU          ?= 2
NUM_THREAD       ?= 16

# -----------------------------------------------------------
# Benchmark Variables:
# -----------------------------------------------------------
OUTPUT_CSV_PATH   ?= benchmark.csv
EVAL_RATIO        ?= 0.2
QUERY_SIZE        ?= 1000
BENCHMARK_PROCESS ?= auto_tune

# -----------------------------------------------------------
# Default Target
# -----------------------------------------------------------
all: help

# -----------------------------------------------------------
# Targets to run embed_dataset.py with different modes
# -----------------------------------------------------------
benchmark:
	python benchmark_index.py \
		--process $(BENCHMARK_PROCESS) \
		--index-dir $(INDEX_OUTPUT_DIR) \
		--emb-path $(EMBEDDING_PATH) \
		--output-csv $(OUTPUT_CSV_PATH) \
		--query-size $(QUERY_SIZE) 
	
index-train:
	python train_index.py \
		--process train \
		--index-base-dir $(INDEX_BASE_DIR) \
		--embedding-path $(EMBEDDING_PATH) \
		--embedding-num $(EMBEDDING_NUM) \
		--index $(INDEX) \
		--train-ratio $(TRAIN_RATIO) \
		--output-prefix $(OUTPUT_PREFIX) \
		--train-jsonl-path $(TRAIN_JSONL_PATH) \
		--num-gpu $(NUM_GPU) \
		--num-thread $(NUM_THREAD)

index-populate:
	python train_index.py \
		--process add \
		--index-base-dir $(INDEX_BASE_DIR) \
		--embedding-path $(EMBEDDING_PATH) \
		--embedding-num $(EMBEDDING_NUM) \
		--index-output-dir $(INDEX_OUTPUT_DIR)

index-all:
	python train_index.py \
		--process all \
		--index-base-dir $(INDEX_BASE_DIR) \
		--embedding-path $(EMBEDDING_PATH) \
		--embedding-num $(EMBEDDING_NUM) \
		--index $(INDEX) \
		--train-ratio $(TRAIN_RATIO) \
		--output-prefix $(OUTPUT_PREFIX) \
		--train-jsonl-path $(TRAIN_JSONL_PATH) \
		--num-gpu $(NUM_GPU) \
		--num-thread $(NUM_THREAD) \
		--index-output-dir $(INDEX_OUTPUT_DIR)

extract-emb:
	python embed_dataset.py \
		--process extract-emb \
		--dataset $(DATASET) \
		--dataset-cache-dir $(DATASET_CACHE_DIR) \
		--dataset-split $(DATASET_SPLIT) \
		--output-emb $(OUTPUT_EMB) \
		--batch $(BATCH)

extract-chunk:
	python embed_dataset.py \
		--process extract-chunk \
		--dataset $(DATASET) \
		--dataset-cache-dir $(DATASET_CACHE_DIR) \
		--dataset-split $(DATASET_SPLIT) \
		--output-chunk $(OUTPUT_CHUNK) \
		--chunk-cpu-ratio $(CHUNK_CPU_RATIO) \
		--batch $(BATCH)

extract-all:
	python embed_dataset.py \
		--process all \
		--dataset $(DATASET) \
		--dataset-cache-dir $(DATASET_CACHE_DIR) \
		--dataset-split $(DATASET_SPLIT) \
		--output-emb $(OUTPUT_EMB) \
		--output-chunk $(OUTPUT_CHUNK) \
		--chunk-cpu-ratio $(CHUNK_CPU_RATIO) \
		--batch $(BATCH)

help:
	@echo "Makefile for FAISS Playground"
	@echo "Available targets:"
	@echo "  benchmark        - Benchmark indexes from directory."
	@echo "  extract-emb      - Extract embeddings from dataset as ndarray."
	@echo "  extract-chunk    - Extract text chunks from dataset as ndarray."
	@echo "  extract-all      - Extract embeddings and text chunks from dataset as ndarray."
	@echo "  index-train      - Train indexes with input options as empty index"
	@echo "  index-populate   - Populate indexes from given dir"
	@echo "  index-all        - Train and populate indexes"
	@echo ""
	@echo "Override variables : "
	@echo "    DATASET, DATASET_CACHE_DIR, DATASET_SPLIT,"
	@echo "    OUTPUT_EMB, OUTPUT_CHUNK, INDEX_TYPE, CHUNK_CPU_RATIO, BATCH"
	@echo "    INDEX_BASE_DIR, EMBEDDING_PATH, EMBEDDING_NUM, INDEX, TRAIN_RATIO,"
	@echo "    OUTPUT_PREFIX, TRAIN_JSONL_PATH, NUM_GPU, NUM_THREAD, INDEX_OUTPUT_DIR"

