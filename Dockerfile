# 使用官方 Python 基础镜像
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# 设置环境变量，防止 Python 写入 .pyc 文件
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 将项目代码复制到容器中
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

## 暴露端口（可选，取决于你的应用）
#EXPOSE 8000

## 设置启动命令
#CMD ["python", "app.py"]
