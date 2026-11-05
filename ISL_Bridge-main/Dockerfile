FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt
RUN pip uninstall -y jax jaxlib || true
COPY . .
CMD ["python", "flask_app/app.py"]