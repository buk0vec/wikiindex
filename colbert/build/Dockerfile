FROM python:3.9-slim

ARG INDEX_NAME

ENV PORT=8893
ENV INDEX_NAME=${INDEX_NAME}
ENV INDEX_ROOT="/colbert/indexes"

WORKDIR /colbert
ADD "build/colbert/collection.tsv" .

WORKDIR /colbert/indexes/${INDEX_NAME}
ADD "build/colbert/index/indexes/${INDEX_NAME}" .

WORKDIR /colbert/checkpoint
ADD "build/colbert/checkpoint" .

EXPOSE ${PORT}

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt install -y g++

WORKDIR /server
COPY server-requirements.txt .
RUN pip install --no-cache-dir -r server-requirements.txt
COPY ./server.py .
CMD ["python3", "server.py"]