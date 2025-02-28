FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 

# allowed generic, avx2, avx512, avx512_spr
ARG FAISS_OPT="avx512"
ARG FAISS_COMMIT="4c315a93"
ARG CMAKE_VERSION="3.30.8"

ENV TZ="Asia/Seoul"
ENV FAISS_COMMIT=4c315a93
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
ENV MKLROOT=/opt/intel/oneapi/mkl/latest
ENV CMAKE_VERSION=3.30.8

WORKDIR /app

RUN apt update &&\
  apt install -y software-properties-common \
  git \
  tzdata \
  gnupg \
  ca-certificates \
  curl \
  libssl-dev \
  &&\
  # TZ config
  ln -fs /usr/share/zoneinfo/$TZ /etc/localtime &&\
  echo $TZ > /etc/timezone &&\
  dpkg-reconfigure --frontend noninteractive tzdata &&\
  # Ubuntu GCC repo
  echo "deb http://archive.ubuntu.com/ubuntu/ bionic main universe" >> /etc/apt/sources.list &&\
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 &&\
  # IntelMKL repo
  curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null &&\
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list &&\
  # python3.11 repo
  add-apt-repository -y "ppa:deadsnakes/ppa" &&\
  apt update && \ 
  apt install -y --no-install-recommends  \
  python3.11 \
  gcc-9 \
  g++-9 \
  intel-oneapi-mkl \
  libomp-dev\
  libgflags-dev\ 
  libc6-dev &&\
  apt-get install -y python3-dev python3.11-dev python3.11-distutils python3.11-venv &&\
  rm -rf /var/lib/apt/lists/*  &&\
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 &&\
  # Link default
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 &&\
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 &&\
  update-alternatives --set gcc /usr/bin/gcc-9 &&\
  update-alternatives --config python3 &&\
  cd /usr/bin &&\
  ln -s python3 python 

# CMake
RUN curl -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz -o cmake-${CMAKE_VERSION}.tar.gz && \
  tar -zxvf cmake-${CMAKE_VERSION}.tar.gz && \
  rm cmake-${CMAKE_VERSION}.tar.gz &&\
  cd cmake-${CMAKE_VERSION} &&\
  ./bootstrap && \
  make -j$(nproc) && \
  make install &&\
  cd - &&\
  rm -rf cmake-${CMAKE_VERSION}

# Build faiss
RUN git clone https://github.com/facebookresearch/faiss && \
  cd faiss && \
  git checkout 4c315a93 && \
  pip install swig &&\
  pip install -i https://pypi.anaconda.org/intel/simple numpy 

RUN cd faiss &&\
  CC=gcc CXX=g++ cmake -B build  . \
  -DPython_EXECUTABLE=$(which python) \
  -DPython_INCLUDE_DIRS=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
  -DPython_LIBRARIES=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -DBUILD_TESTING=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_OPT_LEVEL=${FAISS_OPT} \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON &&\
  if [ "$FAISS_OPT" = "generic" ]; then \
  make -C build faiss; \
  else \
  make -C build faiss_${FAISS_OPT}; \
  fi && \
  make -C build -j swigfaiss &&\
  make -C build install &&\
  cd build/faiss/python && \
  python setup.py install 

CMD ["/bin/bash"]

