FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m dhruv

RUN chown -R dhruv:dhruv /home/dhruv/

COPY --chown=dhruv . /home/dhruv/app/

USER dhruv

RUN cd /home/dhruv/app/ && pip3 install -r requirements.txt

WORKDIR /home/dhruv/app