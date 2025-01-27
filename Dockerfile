# 使用官方 Python 基础镜像
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# 设置环境变量，防止 Python 写入 .pyc 文件
ENV PYTHONUNBUFFERED=1

# 安装 Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh


# 设置环境变量
ENV PATH="/opt/conda/bin:$PATH"

# 更新 Conda 和安装 Python
RUN conda update -n base -c defaults conda && \
    conda create -n myenv python=3.9 && \
    conda clean -a

# 激活 Conda 环境并安装其他依赖
RUN conda install -n myenv pip && \
    conda run -n myenv pip install -r requirements.txt

# 设置工作目录
WORKDIR /app
COPY . /app

## 安装 Python 依赖
#RUN pip install --no-cache-dir -r requirements.txt

## 设置启动命令
#CMD ["python", "app.py"]
