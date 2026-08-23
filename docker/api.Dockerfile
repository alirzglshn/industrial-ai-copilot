FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# CPU-only torch: the default wheels bundle CUDA and are several GB larger for
# no benefit on the hardware this runs on.
RUN pip install --no-cache-dir -e ".[ai]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Cache model weights in the image layer rather than downloading on first
# request, which would otherwise make the first upload appear to hang.
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000

CMD ["uvicorn", "copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
