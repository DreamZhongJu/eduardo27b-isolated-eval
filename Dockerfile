# 阿里云官方 CUDA 开发镜像，模型与评测依赖单独安装。
FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/cuda-profiling:cuda12.8.1-devel-ubuntu24.04@sha256:30e221b7f7bb455656325e69b07a6e7f06b5cf74f265e63d21b91b2799be60c6

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.9.5

# 仅保留阿里云 Ubuntu 软件源，避免基础镜像继承的其他 APT 源。
RUN rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' \
       'deb https://mirrors.aliyun.com/ubuntu/ noble main restricted universe multiverse' \
       'deb https://mirrors.aliyun.com/ubuntu/ noble-updates main restricted universe multiverse' \
       'deb https://mirrors.aliyun.com/ubuntu/ noble-security main restricted universe multiverse' \
       > /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential git curl python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/uv \
    && /opt/uv/bin/python -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ "uv==${UV_VERSION}" \
    && ln -sf /opt/uv/bin/uv /usr/local/bin/uv \
    && /usr/local/cuda/bin/nvcc --version \
    && python3 --version \
    && /opt/uv/bin/uv --version

ENV PATH=/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
