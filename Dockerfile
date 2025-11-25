FROM python:3.12
LABEL authors="stefan"

WORKDIR /app
ADD . /app
RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python", "-m", "your_package"]
#ENTRYPOINT ["top", "-b"]