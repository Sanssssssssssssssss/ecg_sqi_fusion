ARG BASE_IMAGE=ecg-sqi-infer:latest
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY reproduce /opt/reproduce

ENV WFDB_QRS_KIT_FROM_BIN_DIR=/app/tmp/inference_qrs/tools/bin

WORKDIR /opt/reproduce
VOLUME ["/opt/reproduce/work"]

ENTRYPOINT ["python", "run_reproduce.py"]
CMD ["--help"]
