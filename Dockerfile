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

RUN pip install -U pip setuptools wheel

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM reqs

COPY app.py .
COPY models .
COPY tox_predict .

EXPOSE 8080

CMD python app.py