FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.1.0-gpu-py3


RUN apt-get update
RUN apt-get install -y curl unzip python3 python3-setuptools python3-pip python-dev python3-dev python-psutil ffmpeg libsm6 libxext6
RUN pip3 install py4j psutil==5.6.5 numpy==1.17.4
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK 1


# Dumb init
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init

RUN mkdir /opt/app
COPY program /opt/program
RUN chmod +x /opt/program/augmentation.py
ENTRYPOINT ["python3"]
