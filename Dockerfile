# ====== Base Image ======
FROM python:3.11-slim

# ====== Working Directory ======
WORKDIR /app

# ====== Copy and Install Dependencies ======
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ====== Copy Source Code ======
COPY app.py .

# ====== Environment Variables ======
ENV SESSION_TIMEOUT_MINUTES=30

# ====== Expose Streamlit Port ======
EXPOSE 8501

# ====== Run App ======
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
