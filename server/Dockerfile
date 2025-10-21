FROM python:3.10-slim

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  POETRY_CACHE_DIR='/var/cache/pypoetry' \
  POETRY_HOME='/usr/local'

RUN pip install --upgrade pip setuptools wheel
RUN pip --version
RUN pip install poetry

WORKDIR /app
COPY poetry.lock pyproject.toml poetry.toml README.md ./
# COPY requirements.txt .
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction
# RUN pip install -r requirements.txt
COPY main.py modules.py clients.py compiled.json compiled_5.json ./
EXPOSE 8000

CMD ["python", "main.py"]