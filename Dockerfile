FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY analysis.py polars_analysis.py test_analysis.py ./
COPY wine_quality_merged.csv ./

CMD ["python", "analysis.py"]