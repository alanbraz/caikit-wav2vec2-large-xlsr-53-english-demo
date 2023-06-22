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

COPY . .

ENV PROXY_ENDPOINT=${PROXY_ENDPOINT:-"0.0.0.0:8085"}
ENV SERVE_PORT=${GATEWAY_PORT:-8080}

EXPOSE 8080
EXPOSE 8085

CMD python start_runtime.py & ./gateway/app --swagger_path=./gateway/swagger & wait -n
# curl -X POST "https://caikit-example-molformer.fm-model-train-9ca4d14d48413d18ce61b80811ba4308-0000.us-south.containers.appdomain.cloud/v1/caikit.runtime.TextSentiment/TextSentimentService/HuggingFaceSentimentTaskPredict" -H "accept: application/json" -H "grpc-metadata-mm-model-id: text_sentiment" -H "content-type: application/json" -d "{ \"textInput\": { \"text\": \"awful test\" }}"