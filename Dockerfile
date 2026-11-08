# 基础工具镜像：评测代码和模型依赖单独取得与安装。
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.9.5

# Ubuntu 主软件源使用阿里云；基础镜像自带的其他源需由管理员核验。
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g; s@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ "uv==${UV_VERSION}"

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
