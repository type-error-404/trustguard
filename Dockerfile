FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
CMD gunicorn --chdir app --bind 0.0.0.0:$PORT --timeout 180 --workers 1 app:app
