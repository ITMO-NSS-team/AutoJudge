FROM node:22-bookworm-slim AS web-build

WORKDIR /build
COPY src/autojudge/ui/web/package.json src/autojudge/ui/web/package-lock.json ./
RUN npm ci
COPY src/autojudge/ui/web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /data/autojudge \
    && chown -R user:user /data

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTOJUDGE_DATA_DIR=/data/autojudge \
    AUTOJUDGE_WEB_DIST=/home/user/app/src/autojudge/ui/web/dist \
    AUTOJUDGE_SETTINGS_READ_ONLY=1 \
    AUTOJUDGE_AI_ENABLED=1
WORKDIR /home/user/app

COPY --chown=user:user src/autojudge/ui/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

COPY --chown=user:user src/autojudge ./src/autojudge
COPY --from=web-build --chown=user:user /build/dist ./src/autojudge/ui/web/dist

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "src/autojudge/ui/backend", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
