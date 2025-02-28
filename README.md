# FAISS Playground

Project for creating a retrieval system for Retrieval-Augmented Generation (RAG) systems using the FAISS library.

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook Similarity Search library

## Build & Run

### TL;DR

```bash
docker build -f Dockerfile -t faiss:dev0.1 .
docker run faiss:dev0.1
```

### Faiss Installation

#### Requirements

- Intel machine for full Intel MKL support
- Nvidia CUDA (Tested with `CUDA 12.2`)
- Tested with FAISS version [4c315a93](https://github.com/facebookresearch/faiss/commit/4c315a93)

#### Configure FAISS build

```docker
# In Dockerfile line 3, configure FAISS Optimization Level
# Available values
# "generic""   for default mkl
# "avx2"       for mkl AVX2 feature
# "avx512"     for mkl AVX512 feature
# "avx512_spr" for mkl AVX512 Sapphire Rapids
FAISS_OPT="avx512"
```

### Build & Run

```bash
git clone https://github.com/KY00KIM/faiss_playground 
cd faiss_playground

# Build
docker build -f Dockerfile -t faiss:dev0.1 .
docker build -f Dockerfile.prod -t faiss:prod0.1 .

# Run
docker run -it faiss:dev0.1
docker run -it faiss:dev0.1
```
