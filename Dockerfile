FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY rag_pipeline.py .
COPY streamlit_app.py .
COPY data ./data
COPY chroma_db ./chroma_db

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
