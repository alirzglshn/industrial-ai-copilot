FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# CPU-only torch: the default wheels bundle CUDA and are several GB larger for
# no benefit on the hardware this runs on.
RUN pip install --no-cache-dir -e ".[ai]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/app/.hf_cache

# Bake the embedding weights into the image. Without this the model downloads
# on the first upload, which looks like a hang and fails outright offline.
ARG TEXT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV TEXT_EMBEDDING_MODEL=${TEXT_EMBEDDING_MODEL}
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${TEXT_EMBEDDING_MODEL}')"

EXPOSE 8000

CMD ["uvicorn", "copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
