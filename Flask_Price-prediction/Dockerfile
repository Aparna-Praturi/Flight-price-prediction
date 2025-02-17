FROM python:3.10-slim-buster

WORKDIR /app

# Copy requirements first and install
COPY Flask_Price-prediction/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the entire flask-app directory to /app inside the container
COPY ../Flask_Price-prediction /app/Flask_Price-prediction

# Copy py_files and models directories (going up one level)
COPY ../models /app/models
COPY ../py_files /app/py_files
COPY ../data /app/data

# Set environment variables
ENV FLASK_APP=/app/Flask_Price-prediction/app.py 

# Expose port 5000 for the Flask app
EXPOSE 5000


# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

