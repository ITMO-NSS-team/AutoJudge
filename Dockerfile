FROM node:20-slim AS node

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG GITHUB_TOKEN

COPY pyproject.toml README.md ./
COPY src/automas/__init__.py src/automas/__init__.py

RUN if [ -n "$GITHUB_TOKEN" ]; then \
    pip install \
    git+https://${GITHUB_TOKEN}@github.com/ITMO-NSS-team/maseval-research.git; \
    fi

RUN uv pip install --system --no-cache --break-system-packages -e . --group benchmarks

RUN playwright install chromium

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:$PYTHONPATH
