FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    HF_HUB_ENABLE_HF_TRANSFER=1

ENV PATH="${HOME}/miniconda3/bin:${PATH}"
ARG PATH="${HOME}/miniconda3/bin:${PATH}"

RUN mkdir -p /tmp/model && \
    chown -R 1000:1000 /tmp/model && \
    mkdir -p /tmp/data && \
    chown -R 1000:1000 /tmp/data

RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    gcc \
    git \
    locales \
    net-tools \
    wget \
    libpq-dev \
    libsndfile1-dev \
    git \
    git-lfs \
    libgl1 \
    unzip \
    openjdk-11-jre-headless \
    libseccomp-dev \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    git lfs install

WORKDIR /app

RUN mkdir -p /app/.cache
ENV HF_HOME="/app/.cache"
RUN chown -R 1000:1000 /app
USER 1000
ENV HOME=/app

ENV PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && sh Miniconda3-latest-Linux-x86_64.sh -b -p /app/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/app/miniconda/bin:$PATH
ENV CONDA_ENV_MODEL=/app/model_default

RUN conda create -p /app/env -y python=3.12
RUN conda create -p ${CONDA_ENV_MODEL} -y python=3.12

SHELL ["conda", "run","--no-capture-output", "-p","/app/env", "/bin/bash", "-c"]

# RUN conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia && \
#     conda clean -ya && \
#     conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc && conda clean -ya

RUN pip install -U pip
COPY --chown=1000:1000 requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY --chown=1000:1000 . /app/
RUN make sandbox

# give permissions to run sandbox
RUN chmod +x /app/sandbox

ENV PATH="/app:${PATH}"

RUN pip install -e .
# RUN pip install -r requirements_docker.txt

ARG COMP_IMAGE
ENV COMP_IMAGE=$COMP_IMAGE 