FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/output

EXPOSE 8080

CMD ["sh", "-c", "if [ \"$APP_ROLE\" = \"monitor\" ]; then python monitor.py; else gunicorn --bind 0.0.0.0:${PORT:-8080} mobile_dashboard:app; fi"]
