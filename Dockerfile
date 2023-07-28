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

COPY app.py .
COPY speech speech
COPY speech models

EXPOSE 8080

CMD python app.py