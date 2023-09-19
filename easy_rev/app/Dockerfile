FROM python:latest

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

EXPOSE 65432 

CMD ["rev.py"]
