FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates

EXPOSE 3000

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "--timeout", "300", "--graceful-timeout", "30", "-b", "0.0.0.0:3000", "app:app"]