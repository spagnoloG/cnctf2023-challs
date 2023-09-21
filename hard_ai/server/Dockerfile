FROM python:latest

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt install -y wget
RUN wget "https://nextcloud.spanskiduh.dev/s/Q8zWDAtLoAHmJaA/download?path=%2F&files=dataset_val.zip&downloadStartSecret=nhm3n9k9mva" -O dataset_val.zip
RUN unzip dataset_val.zip 

ENTRYPOINT ["python"]

EXPOSE 65432 

CMD ["server.py"]
