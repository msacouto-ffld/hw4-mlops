FROM python:3.10-slim

WORKDIR /app

# Copy requirements first — Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code edit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py .
COPY model/ model/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
