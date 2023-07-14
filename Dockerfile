## before build this image

## generate the protos
# python generate_protos.py

## generate the gateway
# pip install grpc-gateway-wrapper
# rm -rf protos/*training*
# GOOS=linux GOARCH=amd64 grpc-gateway-wrapper \
#         --proto_files protos/*.proto \
#         --metadata mm-model-id \
#         --output_dir gateway \
#         --install_deps
# FROM registry.access.redhat.com/ubi9/ubi-minimal
FROM registry.access.redhat.com/ubi9/python-39 as reqs

USER root

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN yum update -y --allowerasing
# RUN yum install -y nc bind-utils wget
# RUN yum remove *mysql* httpd -y
# RUN find / -name httpd | xargs -n1 rm -rf
RUN yum autoremove -y
RUN yum -y clean all --enablerepo='*'

USER 1001

RUN pip install -U pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM reqs

RUN pip install opencv-python-headless

COPY app.py .
COPY models models
COPY tox_predict tox_predict
COPY tokenizer.py .

EXPOSE 8080

CMD python app.py