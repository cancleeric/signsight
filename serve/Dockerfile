# CPU inference image. Build from the repo root:
#   docker build -f serve/Dockerfile -t signsight .
#   docker run -p 8000:8000 -e SIGNSIGHT_CHECKPOINT=/app/checkpoints/best.pt \
#     -v $(pwd)/checkpoints:/app/checkpoints signsight
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first to keep the image small and the build cacheable.
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r serve/requirements.txt

COPY signsight ./signsight
COPY serve ./serve

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"]
