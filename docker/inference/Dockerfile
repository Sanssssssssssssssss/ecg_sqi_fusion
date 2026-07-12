FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc make libc6-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY docker/inference/constraints.txt ./docker/inference/constraints.txt
COPY src ./src
COPY pretrained/chapter4/seta_e31_leadwise_shared ./pretrained/chapter4/seta_e31_leadwise_shared
COPY pretrained/chapter4/but_e31_query_mean_fused_conformer ./pretrained/chapter4/but_e31_query_mean_fused_conformer
COPY pretrained/inference ./pretrained/inference

RUN python -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch
RUN python -m pip install --no-cache-dir numpy
RUN python -m pip install --no-cache-dir scipy
RUN python -m pip install --no-cache-dir pandas
RUN python -m pip install --no-cache-dir -c docker/inference/constraints.txt scikit-learn
RUN python -m pip install --no-cache-dir matplotlib
RUN python -m pip install --no-cache-dir wfdb PyWavelets \
    "wfdb-qrs-kit @ https://github.com/Sanssssssssssssssss/wfdb-qrs-kit/releases/download/v0.1.0/wfdb_qrs_kit-0.1.0-py3-none-any.whl#sha256=ae2d9622bffa2774499ffc899ae53f127383fdfd6ab231503c06ee543d18ffdd"
RUN python -m pip install --no-cache-dir --no-deps .

RUN python -m src.sqi_pipeline.qrs.setup_paper_detectors \
    --out_dir /app/tmp/inference_qrs/tools

RUN python -c "import hashlib, urllib.request, zipfile; p='/tmp/wfdb.zip'; urllib.request.urlretrieve('https://physionet.org/content/wfdb/get-zip/10.7.0/', p); assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == '632bd8e0e3796a940437d3b7284c228b50a5ab6388f4ad3b84658627a0f819a7'; zipfile.ZipFile(p).extractall('/tmp')" \
    && cd /tmp/wfdb-software-package-10.7.0 \
    && chmod +x configure conf/prompt-c conf/prompt-n conf/which-c install.sh \
    && ./configure --prefix=/usr/local --without-netfiles --without-flac --without-xview \
    && make -C lib -j2 \
    && make -C lib install \
    && ldconfig \
    && rm -rf /tmp/wfdb.zip /tmp/wfdb-software-package-10.7.0

RUN sed -i 's/common_flags = \["gcc", "-O2"\]/common_flags = ["gcc", "-O2", "-std=gnu89"]/' \
    /usr/local/lib/python3.11/site-packages/wfdb_qrs_kit/install.py \
    && python -c "from wfdb_qrs_kit.install import compile_sources; compile_sources('/app/tmp/inference_qrs/tools')" \
    && python -m src.sqi_pipeline.qrs.setup_paper_detectors \
    --out_dir /app/tmp/inference_qrs/tools \
    --no_download \
    --require

RUN python -m src.ecg_sqi_inference verify-bundles

ENTRYPOINT ["python", "-m", "src.ecg_sqi_inference"]
CMD ["--help"]
