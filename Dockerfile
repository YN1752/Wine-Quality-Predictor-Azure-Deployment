FROM python:3.10.18-slim
WORKDIR /app
COPY . /app
RUN pip install -r image_requirements.txt
EXPOSE 8080
CMD ["python", "app.py"]